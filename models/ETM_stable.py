import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence  # rnn 패딩 유틸 추가
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from models.TCN import TemporalBlock


class ETM_Block(nn.Module):
    def __init__(self, config):
        super(ETM_Block, self).__init__()

        self.input_dim = config.enc_in            # D
        self.pred_len = config.pred_len           # T' (예측 길이)
        self.seq_len = config.seq_len             # T (입력 길이)
        self.seg_len = config.seg_len
        self.view_dim = config.view_dim
        self.dropout = config.dropout

        self.embedding_order = nn.Sequential(
            nn.Linear(self.input_dim, self.view_dim),
        )

        self.space_tcn_block = TemporalBlock(n_inputs=self.view_dim, n_outputs=self.view_dim,
                                                kernel_size=3, stride=1, dilation=1, padding=2,
                                                dropout=self.dropout)
        self.event_detection = nn.Sequential(
            nn.Linear(self.view_dim, self.input_dim),
        )
        
        self.sigmoid = nn.Sigmoid()

        self.time_tcn_block = TemporalBlock(
            n_inputs=self.seq_len,
            n_outputs=self.seg_len,
            kernel_size=3, stride=1, dilation=1, padding=2,
            dropout=self.dropout)

        # TCN2DBlockNet 출력 평균 시 shape: [B, T, D]
        # 각 시점별 D차원을 pred_len으로 예측하는 구조로 Linear 정의
        self.time_order_layer = nn.Sequential(
            nn.Linear(self.seg_len, self.pred_len),  # D → D (옵션)
            nn.ReLU(),
            nn.Linear(self.pred_len, self.pred_len),   # D → D
        )


    def forward(self, x, inference=False):
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN encountered in {name}")

        ## 과거 데이터 중에 중요한 부분만 남기기
        # x: [B, T, D]
        order_features = self.embedding_order(x)            # [B, T, D]
        check_nan(order_features, "order_features")
        
        order_features = self.space_tcn_block(order_features.permute(0, 2, 1)).permute(0, 2, 1) # [B,T,V] -> [B,V,T] -> [B,D,T] -> [B,T,D]
        check_nan(order_features, "order_features")
        
        order_features = self.event_detection(order_features)
        check_nan(order_features, "order_features")

        influence_weights = self.sigmoid(order_features)     # [B, T, D]
        check_nan(influence_weights, "influence_weights")

        featured_x = x * influence_weights                    # [B, T, D]
        check_nan(featured_x, "featured_x")

        featured_t = self.time_tcn_block(featured_x).permute(0, 2, 1)  # [B,T,D] -> [B,S,D] -> [B, D, S]
        check_nan(featured_t, "featured_t")

        ## 미래 데이터 값을 예측하기
        future_pattern = self.time_order_layer(featured_t).permute(0, 2, 1) # [B, D, T'] -> [B, T', D]
        check_nan(future_pattern, "future_pattern")
        
        if inference:
            return future_pattern, influence_weights
        else:
            return future_pattern



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.input_dim = config.enc_in
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.seg_len = config.seg_len
        self.d_model = config.d_model
        self.drop_out = config.dropout

        etm_pattern_config = copy.deepcopy(config)
        etm_pattern_config.seq_len = self.seq_len
        etm_pattern_config.pred_len = self.pred_len
        self.etm_pattern = ETM_Block(etm_pattern_config)
        
    def forward(self, past_x, inference=False):
        past_x_last = past_x[:, -1:, :].detach()
        x = past_x - past_x_last

        if inference:
            result, pattern_mask = self.etm_pattern(x, inference=inference)
            return result + past_x_last, pattern_mask
        else:
            result = self.etm_pattern(x, inference=inference)
            return result + past_x_last