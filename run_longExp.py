import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.utils import fix_seed
import traceback
import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='SegRNN_720_720', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # SegRNN
    parser.add_argument('--rnn_type', default='gru', help='rnn_type')
    parser.add_argument('--dec_way', default='pmf', help='decode way')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length')
    parser.add_argument('--win_len', type=int, default=48, help='windows length')
    parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    # parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # ETM
    parser.add_argument('--view_dim', type=int, default=64, help='local info hidden size')
    parser.add_argument('--final_hidden', type=int, default=64, help='output hidden size')
    parser.add_argument('--ths', type=float, default=0.5, help='ths of event occur')
    parser.add_argument('--trend_weight', type=float, default=0.1, help='trend importance (0~1)')
    parser.add_argument('--num_tcn_layers', type=int, default=3, help='num of tcn_layers (1~10)')
    parser.add_argument('--exp_len', type=int, default=512, help='expand dim length')


    # optimization
    parser.add_argument('--loss', type=str, default='mae', help='loss function',
                        choices=['mae', 'mse', 'hetero', 'nll', 'mae_l1_penalty', 'spike_mae', 'rae', 'rmsle', 'nrmse'])
    parser.add_argument('--scheduler', type=str, default='plateau', help='learning rate scheduler', 
                        choices=['plateau', 'cosine', 'onecycle', 'cosine_annealing'])
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--desc', type=str, default='', help='exp description')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    
    parser.add_argument('--pct_start', type=float, default=0.5, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    # ETM SY
    parser.add_argument('--decomp_thershold', type=float, default=1.5, help='decomposition threshold')
    parser.add_argument('--h_len', type=int, default=96, help='horizon length for ETM SY')
    parser.add_argument('--h_ch', type=int, default=64, help='hidden channels for TCN')
    parser.add_argument('--reduction', type=int, default=1, help='reduction for TCN')
    parser.add_argument('--rank', type=int, default=4, help='rank for TCN')
    parser.add_argument('--min_kernel', type=int, default=3, help='minimum kernel size for TCN')
    parser.add_argument('--max_kernel', type=int, default=25, help='maximum kernel size for TCN')

    # ablation
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size for TCN')
    parser.add_argument('--dilation', type=int, default=1, help='dilation size for TCN')
    
    args = parser.parse_args()    

    # random seed
    seed = args.random_seed
    fix_seed(seed)

    print('Args in experiment:')

    Exp = Exp_Main

    flag = False

    dt = datetime.datetime.now().strftime('%m%d-%H%M')

    # OOM이 발생할 경우, Batch size를 줄여서 다시 실행
    while not flag:
        try:
            if args.is_training:
                for ii in range(args.itr):
                    # setting record of experiments
                    setting = '{}/{}_{}_ft{}_bs{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
                        args.model,
                        dt,
                        args.model_id,
                        args.features,
                        args.batch_size,
                        args.seq_len,
                        args.pred_len,
                        args.d_model,
                        args.dropout,
                        args.rnn_type,
                        args.dec_way,
                        args.seg_len,
                        args.loss,
                        args.desc,ii)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting), flush=True)
                    exp.train(setting)

                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)

                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)

                    torch.cuda.empty_cache()
            else:
                ii = 0
                setting = '{}/{}_{}_ft{}_bs{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
                    args.model,
                    dt,
                    args.model_id,
                    args.features,
                    args.batch_size,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.dropout,
                    args.rnn_type,
                    args.dec_way,
                    args.seg_len,
                    args.loss,
                    args.desc, ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()
            flag = True
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("OOM error occurred, reducing batch size and retrying...")
                traceback.print_exc()

                if args.batch_size <= 1:
                    print("Batch size is already at minimum, cannot reduce further.")
                    traceback.print_exc()
                    break
                args.batch_size = max(1, args.batch_size // 2)
            else:
                print("An unexpected error occurred:")
                traceback.print_exc()
                break
