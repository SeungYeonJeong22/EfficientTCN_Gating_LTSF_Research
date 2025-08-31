from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import EfficientTCN, Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, VanillaRNN, SegRNN, ETM_stable, ETM_mix, ETM_sota, ETM_sota2, TimesNet, MICN, ETM, ETM_SJ, FEDformer
from models.ablation import ETM_ab1, ETM_ab2, ETM_ab3
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, CosineAnnealingWarmUpRestarts
from utils.metrics import metric
from utils.utils import save_plot_weights

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
import shutil

import warnings
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


warnings.filterwarnings('ignore')


# ---------- RMSLE ----------
def rmsle_torch(y_pred: torch.Tensor,
                y_true: torch.Tensor,
                reduce_over=None,
                eps: float = 1e-12,
                nonneg_mode: str = "clamp"):
    """
    RMSLE = sqrt( mean( (log(1+y_true) - log(1+y_pred))^2 ) )
    - 표준 정의는 y >= 0 가정. 실전에서 음수가 있을 수 있어 기본은 0으로 clamp.
      nonneg_mode='shift' 로 두면 (min값을 0으로 맞추는) 자동 시프트 사용.
    """
    if reduce_over is None:
        reduce_over = list(range(y_true.ndim))

    if nonneg_mode == "shift":
        m = torch.minimum(y_true.min(), y_pred.min())
        shift = (-m + 1e-6) if (m < 0) else 0.0
        yt = y_true + shift
        yp = y_pred + shift
    elif nonneg_mode == "clamp":
        yt = torch.clamp(y_true, min=0.0)
        yp = torch.clamp(y_pred, min=0.0)
    else:
        raise ValueError("nonneg_mode must be 'clamp' or 'shift'.")

    diff = torch.log1p(yt + eps) - torch.log1p(yp + eps)
    mse = torch.mean(diff**2, dim=reduce_over)
    return torch.sqrt(mse + eps)

# ---------- NRMSE (mean 기준 정규화) ----------
def nrmse_torch(y_pred: torch.Tensor,
                y_true: torch.Tensor,
                reduce_over=None,
                norm: str = "mean",
                eps: float = 1e-12):
    """
    그림의 정의: NRMSE = RMSE / \bar{y}
      - 'mean'  : 분모 = |mean(y_true)|   (평균이 0 근처일 때 폭주 방지 위해 절대값)
      - 'std'   : 분모 = std(y_true)      (대안)
      - 'range' : 분모 = (max - min)      (대안)
    """
    if reduce_over is None:
        reduce_over = list(range(y_true.ndim))

    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=reduce_over) + eps)

    if norm == "mean":
        denom = torch.mean(y_true, dim=reduce_over).abs()
    elif norm == "std":
        denom = torch.std(y_true, dim=reduce_over)
    elif norm == "range":
        # reduce_over가 여러 축일 경우 전체 텐서에서 범위 계산
        denom = (y_true.max() - y_true.min())
    else:
        raise ValueError("norm must be one of {'mean','std','range'}.")

    return rmse / (denom + eps)


def rae_torch(y_pred: torch.Tensor, y_true: torch.Tensor, reduce_over=None, eps: float = 1e-12):
    """
    y_* shape 예: [B, T, C] 또는 [N]
    reduce_over: 합산할 차원 리스트. 기본(None)이면 모든 차원에 대해 RAE(스칼라).
                 예) 각 배치마다 계산: reduce_over=[1,2]  (B는 남김)
    반환: 스칼라 또는 배치별 텐서
    """
    if reduce_over is None:
        reduce_over = list(range(y_true.ndim))  # 전 차원 합
    num = torch.sum(torch.abs(y_true - y_pred), dim=reduce_over)
    y_bar = torch.mean(y_true, dim=reduce_over, keepdim=True)
    den = torch.sum(torch.abs(y_true - y_bar), dim=reduce_over)
    return num / (den + eps)



alpha = 0.5   # spike 구간 가중
beta  = 0.1   # 게이트 정렬
rho   = 0.01  # 게이트 시간적 안정화
eta   = 1e-4  # 스파스


# ---------- 2) Spike-weighted MAE ----------
def spike_weight_from_y(y_true, win=24, z=1.5, eps=1e-6):
    """
    y_true: [B,P,C]  → w: [B,P,C] in [0,1]
    간단한 롤링 평균/표준편차로 z-score 스파이크 검출 (causal 근사)
    """
    B, P, C = y_true.shape
    y = y_true.permute(0, 2, 1)                   # [B,C,P]
    k = min(win, P)
    pad = (k - 1, 0)
    avg = F.avg_pool1d(F.pad(y, pad, mode='replicate'), kernel_size=k, stride=1)
    sqr = F.avg_pool1d(F.pad(y**2, pad, mode='replicate'), kernel_size=k, stride=1)
    var = (sqr - avg**2).clamp_min(0.0)
    std = (var + eps).sqrt()

    zscore = (y - avg).abs() / (std + eps)        # [B,C,P]
    # hard + soft 결합 (부드럽게)
    soft = torch.sigmoid(2.0 * (zscore - z))      # (0,1)
    w = soft                                      # 필요시 hard=(zscore>z).float()와 max 사용
    return w.permute(0, 2, 1)                     # [B,P,C]

def weighted_mae(pred, target, w, eps=1e-6):
    return (w * (pred - target).abs()).sum() / w.sum().clamp_min(eps)

# ---------- 3) Gate alignment (L1) ----------
def enc_spike_mask_from_x(x, z=1.0, pool_k=3, eps=1e-6):
    """
    x: [B,L,C] → m: [B,1,L] (채널 평균된 급변 마스크)
    """
    x_ch = x.permute(0, 2, 1)                      # [B,C,L]
    prev = F.pad(x_ch, (1,0), mode='replicate')[:, :, :-1]
    d = (x_ch - prev).abs()                        # [B,C,L]
    if pool_k > 1:
        d = F.avg_pool1d(d, kernel_size=pool_k, stride=1, padding=pool_k//2)
    d_mean = d.mean(dim=1, keepdim=True)           # [B,1,L]
    mu = d_mean.mean(dim=2, keepdim=True)
    sd = d_mean.std(dim=2, keepdim=True) + eps
    zscore = (d_mean - mu) / sd
    m = torch.sigmoid(2.0 * (zscore - z))          # [B,1,L] in (0,1)
    return m

def gate_alignment_loss_L1(model, x):
    """
    model.multiviewTCN.view_blocks 안의 last_g1/last_g2와
    입력기반 마스크를 L1로 정렬
    """
    m = enc_spike_mask_from_x(x.detach())          # [B,1,L]
    loss, n = 0.0, 0
    for blk in model.multiviewTCN.view_blocks:
        g1 = getattr(blk, "last_g1", None)         # [B,hid,L]
        g2 = getattr(blk, "last_g2", None)
        if g1 is None or g2 is None: 
            continue
        M = m.expand_as(g1)
        loss += F.l1_loss(g1, M) + F.l1_loss(g2, M)
        n += 2
    return loss / max(n, 1)

# ---------- 4) 정규화 ----------
def temporal_tv(g):
    return (g[:, :, 1:] - g[:, :, :-1]).abs().mean()

def gate_regularizers(model):
    tv, n = 0.0, 0
    for blk in model.multiviewTCN.view_blocks:
        if hasattr(blk, "last_g1"): tv += temporal_tv(blk.last_g1); n += 1
        if hasattr(blk, "last_g2"): tv += temporal_tv(blk.last_g2); n += 1
    tv = tv / max(n, 1)
    l1 = sum(blk.l1_penalty() for blk in model.multiviewTCN.view_blocks)
    return tv, l1


class HeteroLoss(nn.Module):
    def __init__(self, eps: float = 1):
        super().__init__()
        self.eps = eps

    def forward(self, outputs: torch.Tensor, var: torch.Tensor, target: torch.Tensor):
        # var가 너무 작아지지 않도록 클램핑
        v = var.clamp(min=self.eps)                             # (B,P,C)
        loss = 0.5 * (torch.abs(outputs - target) / v + torch.log(v))
        return loss.mean()
    
class HuberHeteroLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.eps   = eps

    def forward(self, mu, logvar, target):
        v = torch.exp(logvar).clamp(min=self.eps)
        r = mu - target
        # Huber residual term
        abs_r = torch.abs(r)
        hub = torch.where(abs_r <= self.delta,
                          0.5 * r**2,
                          self.delta * (abs_r - 0.5*self.delta))
        # hetero NLL
        nll = hub / v + 0.5 * torch.log(v)
        return nll.mean()
    

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # self.writer = SummaryWriter(log_dir="runs/experiment_ETM_sy", flush_secs=5)


    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'VanillaRNN': VanillaRNN,
            'SegRNN': SegRNN,
            'ETM_stable': ETM_stable,
            'ETM_sy' : EfficientTCN,
            'ETM_mix' : ETM_mix,
            'ETM_sota': ETM_sota,
            'ETM_sota2': ETM_sota2,
            'TimesNet': TimesNet,
            'MICN': MICN,
            'ETM' : ETM,
            'ETM_ab1' : ETM_ab1,
            'ETM_ab2' : ETM_ab2,
            'ETM_ab3' : ETM_ab3,
            'ETM_SJ' : ETM_SJ,
            'FEDformer': FEDformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
            # model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        print('Use device: {}'.format(self.device))

        model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "hetero":
            criterion = HeteroLoss()
        elif self.args.loss == "huber_hetero":            
            criterion = HuberHeteroLoss()
        elif self.args.loss == 'nll':
            criterion = nn.GaussianNLLLoss(reduction='mean')
        elif self.args.loss == 'rae':
            criterion = rae_torch
        elif self.args.loss == 'rmsle':
            criterion = rmsle_torch
        elif self.args.loss == 'nrmse':
            criterion = nrmse_torch

        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero"):
                            outputs, var = self.model(batch_x)  
                            var =  var.to(self.device)                                              
                        elif any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                              # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                              outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero" or self.args.loss=="nll"):
                        outputs, var = self.model(batch_x)    
                        var =  var.to(self.device)          
                    elif any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()


                if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero"):
                    loss = criterion(outputs, var, batch_y)
                elif self.args.loss == 'nll':
                    loss = criterion(outputs, batch_y, var)
                elif self.args.loss == 'spike_mae':
                    base_mae  = F.l1_loss(outputs, batch_y)
                    w         = spike_weight_from_y(batch_y, win=24, z=1.5)
                    spike_mae = weighted_mae(outputs, batch_y, w)

                    L_gate = gate_alignment_loss_L1(self.model, batch_x)  # x:[B,L,C]
                    tv, l1 = gate_regularizers(self.model)
                    loss = base_mae + alpha*spike_mae + beta*L_gate + rho*tv + eta*l1                                            
                else:
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # scheduler
        if self.args.scheduler.lower() == 'onecycle':
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate,
                                                anneal_strategy='linear')
        elif self.args.scheduler.lower() == 'plateau':            
            scheduler = lr_scheduler.ReduceLROnPlateau(
                            optimizer=model_optim,
                            mode='min',
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6
                        )
        elif self.args.scheduler.lower() == 'cosine':
            scheduler = lr_scheduler.CyclicLR(
                            optimizer=model_optim,
                            base_lr=1e-5,
                            max_lr=1e-3,
                            step_size_up=2000,
                            mode='triangular2'
                        )
        elif self.args.scheduler.lower() == 'cosine_annealing':
            scheduler = CosineAnnealingWarmUpRestarts(model_optim, T_0=10, T_mult=1, eta_max=1e-4,  T_up=10, gamma=0.5)


        weights_save_dir_name = f"weights/{setting}"
        if os.path.exists(weights_save_dir_name):
            shutil.rmtree(weights_save_dir_name, ignore_errors=True)  # 기존 폴더 삭제


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero"):
                            outputs, var = self.model(batch_x)  
                            var =  var.to(self.device)                      
                        elif any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                              # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                              outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero"):
                            loss = criterion(outputs, var, batch_y)
                        else:
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else: 
                    if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero" or self.args.loss=="nll"):
                        outputs, var = self.model(batch_x)
                        var =  var.to(self.device)                  
                    elif any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]                        
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.loss == "mae_l1_penalty":
                        l1_reg = 0.0
                        for block in self.model.multiviewTCN.view_blocks:
                            l1_reg += block.l1_penalty()


                    if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero"):
                        loss = criterion(outputs, var, batch_y)
                    elif self.args.loss == "mae_l1_penalty":
                        loss = criterion(outputs, batch_y) + l1_reg
                    elif self.args.loss == 'nll':
                        loss = criterion(outputs, batch_y, var)
                    elif self.args.loss == 'spike_mae':
                        base_mae  = F.l1_loss(outputs, batch_y)
                        w         = spike_weight_from_y(batch_y, win=24, z=1.5)
                        spike_mae = weighted_mae(outputs, batch_y, w)

                        L_gate = gate_alignment_loss_L1(self.model, batch_x)  # x:[B,L,C]
                        tv, l1 = gate_regularizers(self.model)
                        loss = base_mae + alpha*spike_mae + beta*L_gate + rho*tv + eta*l1                        
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    # print("Epoch: {}, Step: {}/{} | Loss: {:.7f}".format(
                    #     epoch + 1, i + 1, train_steps, loss.item()))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            if not os.path.exists(path):
                os.makedirs(path)
                            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


            # with torch.no_grad():
            #     scale_w = self.model.net.scale_weight       # (K,)
            #     scale_p = torch.softmax(scale_w / 0.7, dim=0).cpu().numpy() # 0.7: softmax temperature
            # TensorBoard에 기록
            # self.writer.add_histogram("scale_weights/raw", scale_w.detach().cpu().numpy(), epoch)
            # self.writer.add_histogram("scale_weights/softmax", scale_p, epoch)
            # self.writer.add_scalar("loss/train", train_loss, epoch)
            # self.writer.add_scalar("loss/val",   vali_loss,   epoch)
            # self.writer.add_scalar("lr",         model_optim.param_groups[0]["lr"], epoch)
            
            # alpha 트래킹
            # for name, module in self.model.named_modules():
            #     if isinstance(module, ResidualTCN):
            #         print(f"alpha/{name}", module.alpha.item())
            #         self.writer.add_scalar(f"alpha/{name}", module.alpha.item())

            # for name, param in self.model.named_parameters():
            #     f_name_base=f"weights/{setting}/epoch_{epoch + 1}"
            #     if self.args.model=="ETM_sy":
            #         tgt_layer = ['tcn1', 'tcn2', 'attn', 'final_fc']
            #         for l_name in tgt_layer:
            #             f_name = f_name_base + f"/{l_name}.png"
            #             if name.__contains__(l_name):
            #                 save_plot_weights(
            #                     f_name=f_name,
            #                     model=self.model,
            #                     l_name=l_name,
            #                     tsne_perplexity=20,
            #                     tsne_iter=1000
            #                 )

            #     elif self.args.model == "ETM_stable":
            #         tgt_layer = ['feature_dynamics_encoding', 'event_detection', 'time_tcn_block', 'time_gate_layer', 'prediction_time']
            #         for l_name in tgt_layer:
            #             f_name = f_name_base + f"/{l_name}.png"
            #             if name.__contains__(l_name):
            #                 save_plot_weights(
            #                     f_name=f_name,
            #                     model=self.model,
            #                     l_name=l_name,
            #                     tsne_perplexity=20,
            #                     tsne_iter=1000
            #                 )

            #     elif self.args.model == "ETM_mix":
            #         tgt_layer = ['feature_dynamics_encoding', 'event_detection', 'time_tcn_block', 'time_gate_layer', 'prediction_time']
            #         for l_name in tgt_layer:
            #             f_name = f_name_base + f"/{l_name}.png"
            #             if name.__contains__(l_name):
            #                 save_plot_weights(
            #                     f_name=f_name,
            #                     model=self.model,
            #                     l_name=l_name,
            #                     tsne_perplexity=20,
            #                     tsne_iter=1000
            #                 )                            


        # self.writer.flush()
        # self.writer.close()

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(early_stopping.best_model_state)

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # if test:
            # print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        begin_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.loss=="hetero" or self.args.loss=="huber_hetero":
                            outputs, var = self.model(batch_x)  
                            var =  var.to(self.device)                      
                        elif any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model=="ETM_sy" and (self.args.loss=="hetero" or self.args.loss=="huber_hetero" or self.args.loss=="nll"):
                        outputs, var = self.model(batch_x)  
                        var =  var.to(self.device)
                    elif any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]                        
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                # if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        ms = (time.time() - begin_time) * 1000 / len(test_data)

        # if self.args.test_flop:
        #     test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
        #     exit()

        # fix bug
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + "_".join(setting.split("_")[2:]) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, ms/sample:{}'.format(mse, mae, ms))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, ms/sample:{}'.format(mse, mae, ms))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST', 'ETM'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
