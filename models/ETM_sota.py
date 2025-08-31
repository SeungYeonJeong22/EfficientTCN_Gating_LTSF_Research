import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence  # rnn 패딩 유틸 추가
import copy
import numpy as np
from models.TCN import TemporalConvNet

class ETM_Block(nn.Module):
    def __init__(self, config, kernel_size=1):
        super(ETM_Block, self).__init__()

        self.input_dim = config.enc_in
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.seg_len = config.seg_len
        self.view_dim = config.view_dim
        self.num_tcn_layers = config.num_tcn_layers
        self.drop_out = config.dropout
        self.kernel_size = kernel_size
        
        self.position_embedding = nn.Parameter(torch.randn(1, config.seq_len, config.enc_in))
                
        # Event Detection: (batch, seq_len, d_model) -> (batch, seq_len, input_dim)
        self.event_detection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.view_dim),
            nn.ReLU(),
            nn.Linear(self.view_dim, self.input_dim),
        )
        
        '''self.event_score = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.view_dim),
            nn.ReLU(),
            nn.Linear(self.view_dim, 1)
        )'''

        num_channels = [self.seg_len] * self.num_tcn_layers
        self.time_tcn_block = TemporalConvNet(self.seq_len, num_channels, kernel_size=self.kernel_size, dropout=self.drop_out)

        self.prediction_time = nn.Sequential(
            nn.Linear(self.seg_len, self.seg_len // 2),
            nn.ReLU(),
            nn.Linear(self.seg_len // 2, self.pred_len)
        )


    def forward(self, prev, inference=False, absolute_start=True, absolute_final=True):
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN encountered in {name}")

        # 🔹 origin 원점으로
        if absolute_start:
            seq_last = prev[:, -1:, :].detach()
            x = prev
        else:
            seq_last = prev[:, -1:, :].detach()
            x = prev - seq_last
            
        x = x + self.position_embedding

        trigger_detection = self.event_detection(x)
        trigger_detection = torch.sigmoid(x + trigger_detection) # Residual connection O
        check_nan(trigger_detection, "trigger_detection after sigmoid with event_detection")
        
        # trigger_detection = torch.sigmoid(self.event_score(x))
        featured_x = x * trigger_detection
        check_nan(featured_x, "featured_x after indexing")

        # TimeResidualTCNBlock 이후
        time_generated = self.time_tcn_block(featured_x)
        check_nan(time_generated, "time_generated after time_tcn_block")

        # 최종 Prediction
        result = self.prediction_time(time_generated.permute(0, 2, 1)).permute(0, 2, 1)
        check_nan(result, "result after prediction_time")

        if inference:
            if absolute_final:
                return result, trigger_detection
            else:
                return result + seq_last, trigger_detection
        else:
            if absolute_final:
                return result
            else:
                return result + seq_last
            
        
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.input_dim = config.enc_in
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.seg_len = config.seg_len
        self.d_model = config.d_model
        self.final_hidden = config.final_hidden
        self.drop_out = config.dropout

        valid_pairs, valid_refer = self.get_valid_kernel_dilation(self.seq_len)

        # ETM Block 구성
        k, d = valid_pairs[-1]
        d = valid_refer[k][0]
        normal_config_long = copy.deepcopy(config)
        normal_config_long.seq_len = self.seq_len
        normal_config_long.pred_len = self.pred_len
        self.normal_etm_block_pattern = ETM_Block(normal_config_long, kernel_size=k)


    @staticmethod
    def get_valid_kernel_dilation(seq_len, max_kernel_size=20, max_dilation=128):
        valid_pairs, valid_refer = [], dict()
        for k in range(3, max_kernel_size + 1, 2):
            for d in range(1, max_dilation + 1):
                receptive_field = 1 + (k - 1) * d
                if receptive_field >= seq_len:
                    padding = (k - 1) * d / 2
                    if padding.is_integer():
                        valid_pairs.append((k, d))
                        valid_refer.setdefault(k, []).append(d)
                        break
        return valid_pairs, valid_refer

    def forward(self, prev, inference=False):
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN encountered in {name}")
    
        # �뵻 origin �먯젏�쇰줈 鍮쇨린 O (媛� 媛먯냼 * 1)
        seq_last = prev[:, -1:, :].detach()
        x = prev - seq_last

        # inference=True�� 寃쎌슦�� 3媛� 諛섑솚 留욎쓬
        if inference:
            pattern, pattern_mask = self.normal_etm_block_pattern(x, inference=True)
            return pattern + seq_last, pattern_mask
        else:
            pattern = self.normal_etm_block_pattern(x, inference=False)
            return pattern + seq_last