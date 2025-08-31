import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# ---------------- Utilities ----------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class SepConv1d(nn.Module):
    """Depthwise + pointwise separable Conv1d with weight norm."""
    def __init__(self, in_ch, out_ch, kernel_size, padding, dilation):
        super().__init__()
        depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=padding, dilation=dilation,
            groups=in_ch, bias=False
        )
        pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)
        self.depthwise = weight_norm(depthwise, name='weight', dim=1)
        self.pointwise = weight_norm(pointwise, name='weight', dim=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# ---------------- Gating modules ----------------
class ChannelSEGate(nn.Module):
    """
    Classic SE-style channel gate.
    If act='sigmoid' (default), range in [0,1].
    If act='tanh01', applies 0.5*(tanh(x)+1) to keep [0,1] but zero-centered gradients.
    """
    def __init__(self, ch, reduction=4, act: str = "sigmoid"):
        super().__init__()
        self.fc1 = nn.Conv1d(ch, max(1, ch // reduction), 1)
        self.fc2 = nn.Conv1d(max(1, ch // reduction), ch, 1)
        self.act = act

    def forward(self, x):  # x: [B, C, L]
        w = x.mean(dim=2, keepdim=True)
        w = F.relu(self.fc1(w))
        w = self.fc2(w)
        if self.act == "sigmoid":
            w = torch.sigmoid(w)
        elif self.act == "tanh01":
            w = 0.5 * (torch.tanh(w) + 1.0)
        elif self.act == "hardsigmoid":
            w = F.hardsigmoid(w)  # piecewise-linear, sharper saturation
        else:
            raise ValueError(f"Unknown act: {self.act}")
        return x * w, w

class TemporalSEGate(nn.Module):
    """
    Temporal squeeze-excitation: produce a gate for every timestep (and channel).
    Captures local bursts/spikes better than pure channel SE.
    """
    def __init__(self, ch, k: int = 7):
        super().__init__()
        pad = (k - 1) // 2
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=True)

    def forward(self, x):  # x: [B,C,L]
        g = F.relu(self.dw(x))
        g = torch.sigmoid(self.pw(g))
        return x * g, g

class SpikeGate(nn.Module):
    """
    High-pass (derivative) driven sparse gate to emphasize spikes.
    Gate: sigmoid(gamma * (pool(|x(t)-x(t-1)|) - tau))
    - gamma and tau are learnable per-channel scalars.
    - We keep the conv fixed to [1,-1] to behave like a derivative.
    """
    def __init__(self, ch, pool_k: int = 3):
        super().__init__()
        # length-preserving HPF: kernel=3, padding=1
        self.hpf = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        with torch.no_grad():
            w = torch.zeros(ch, 1, 3)
            w[:, 0, 0] = -1.0
            w[:, 0, 1] =  0.0
            w[:, 0, 2] =  1.0
            self.hpf.weight.copy_(w)
        for p in self.hpf.parameters():
            p.requires_grad = False  # keep fixed

        self.pool_k = pool_k
        # broadcast-friendly parameters
        self.gamma = nn.Parameter(torch.ones(1, ch, 1))
        self.tau   = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x):  # x: [B, C, L]
        d = torch.abs(self.hpf(x))  # [B, C, L]
        if self.pool_k > 1:
            d = F.avg_pool1d(d, kernel_size=self.pool_k, stride=1, padding=self.pool_k // 2)
        g = torch.sigmoid(self.gamma * (d - self.tau))  # [B, C, L]
        return x * g, g

class GLUGate(nn.Module):
    """
    Gated Linear Unit style gate at each timestep.
    Performs: y = (W*x) * sigmoid(V*x).
    """
    def __init__(self, ch):
        super().__init__()
        self.lin = nn.Conv1d(ch, ch, 1)
        self.gate = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        sig = torch.sigmoid(self.gate(x))
        return self.lin(x) * sig, sig

# ---------------- Patched GatedTCNBlock ----------------
class ETMBlock(nn.Module):
    """
    TCN block with selectable gating:
      - gating_mode in {'se', 'tse', 'spike', 'glu'}
      - gate_act in {'sigmoid', 'tanh01', 'hardsigmoid'} for 'se' mode
    """
    def __init__(
        self, in_ch, hid_ch, kernel_size, dilation,
        dropout=0.1, reduction=2, l1_lambda=1e-4,
        gating_mode: str = "se", gate_act: str = "sigmoid"
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.l1_lambda = l1_lambda
        self.gating_mode = gating_mode

        # conv stack
        self.conv1 = SepConv1d(in_ch, hid_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        # self.norm1 = nn.BatchNorm1d(hid_ch)
        self.norm1 = nn.LayerNorm(hid_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = SepConv1d(hid_ch, hid_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        # self.norm2 = nn.BatchNorm1d(hid_ch)
        self.norm2 = nn.LayerNorm(hid_ch)
        self.drop2 = nn.Dropout(dropout)

        # gates
        if gating_mode == "se":
            self.gate1 = ChannelSEGate(hid_ch, reduction=reduction, act=gate_act)
            self.gate2 = ChannelSEGate(hid_ch, reduction=reduction, act=gate_act)
        elif gating_mode == "tse":
            self.gate1 = TemporalSEGate(hid_ch, k=7)
            self.gate2 = TemporalSEGate(hid_ch, k=7)
        elif gating_mode == "spike":
            self.gate1 = SpikeGate(hid_ch, pool_k=3)
            self.gate2 = SpikeGate(hid_ch, pool_k=3)
        elif gating_mode == "glu":
            self.gate1 = GLUGate(hid_ch)
            self.gate2 = GLUGate(hid_ch)
        else:
            raise ValueError(f"Unknown gating_mode: {gating_mode}")

        self.ch_expand = nn.Conv1d(in_ch, hid_ch, 1)
        self.l1_weight = None

    def forward(self, x):  # x: [B, L, C]
        resid = self.ch_expand(x.permute(0, 2, 1))  # [B,hid,L]

        out = self.conv1(x.permute(0, 2, 1))
        out = self.chomp1(out)
        out = self.norm1(out.permute(0, 2, 1))  # [B, L, hid]
        out = F.relu(out)
        out = self.drop1(out.permute(0, 2, 1))  # back to [B, hid, L]

        # out, g1 = self.gate1(out)  # gate #1

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out.permute(0, 2, 1))  # [B, L, hid]
        out = F.relu(out)
        out = self.drop2(out.permute(0, 2, 1))  # back to [B, hid, L]

        out, g2 = self.gate2(out)  # gate #2

        # # track L1 penalty on gates to encourage sparsity
        # with torch.no_grad():
        #     g_avg = 0.5 * (g1.abs().mean() + g2.abs().mean())
        # self.l1_weight = g_avg

        out = out + resid  # residual
        return out.permute(0, 2, 1)  # back to [B,L,C]

    def l1_penalty(self):
        if self.l1_weight is None:
            return torch.tensor(0.0, device=self.conv1.depthwise.weight.device)
        return self.l1_lambda * self.l1_weight

class ETMNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_ch, h_ch = config.enc_in, 36
        seq_len = config.seq_len

        # dropout = config.dropout
        dropout = 0.25
        views = [(9,3)]

        # 뷰별 블록을 ModuleList에 저장
        self.view_blocks = nn.ModuleList([
            ETMBlock(in_ch, h_ch, k, d,
                          dropout=dropout,
                          l1_lambda=1e-4,
                          reduction=4,
                          gating_mode="se", gate_act="tanh01")
            for k, d in views
        ])
        self.drop_rate = dropout

        # 뷰 융합용 1×1 Conv
        self.fusion_conv = nn.Conv1d(len(views)*h_ch, h_ch, kernel_size=1)

        self.downsample = nn.AdaptiveAvgPool1d(seq_len // 2)
        self.upsamle = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # 예측 헤드
        self.pred_channel = nn.Sequential(
            nn.Linear(h_ch, h_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_ch, in_ch)
        )

        self.pred_time = nn.Sequential(
            nn.Linear(config.seq_len, config.pred_len),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.pred_len, config.pred_len)
        )

        self.var_time = nn.Sequential(
             nn.Linear(config.seq_len, config.pred_len),
             nn.ReLU(),
             nn.Dropout(dropout),
             nn.Linear(config.pred_len, config.pred_len),
             nn.Softplus()
        )

    def forward(self, x):
        B, L, C = x.shape

        feats = []
        for block in self.view_blocks:
            x_feat = self.downsample(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L/2, C]
            x_feat = block(x_feat)
            x_feat = self.upsamle(x_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, C]
            feats.append(x_feat)  # [B, L, C] for each view
            # feats.append(block(x))


        
        # 1) 뷰별 출력 채널 병합
        fusion = torch.cat(feats, dim=2).permute(0, 2, 1) # [B, L, C*len(views)]
        # 2) 1×1 Conv로 차원 축소
        fusion = self.fusion_conv(fusion)  # [B, L, C*len(views)]
        # 3) 예측 헤드 적용
        j_feat = fusion.permute(0, 2, 1)      # [B, L, h_ch]
        ch_out = self.pred_channel(j_feat)  # [B, L, C]
        ch_out = ch_out.permute(0, 2, 1)      # [B, C, L]

        out_feat = x.permute(0, 2, 1) + ch_out
        
        out = self.pred_time(out_feat).permute(0, 2, 1)           # [B, P, C]
        var  = self.var_time(out_feat).permute(0, 2, 1)           # [B, P, C]

        return out, var


# Wrapper Model
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.etm_nets = ETMNetwork(config)

    def forward(self, x):  # x: [B, L, C]
        past_x_last = x[:, -1:, :].detach()
        x = x - past_x_last

        result, var = self.etm_nets(x)

        return result + past_x_last, var
