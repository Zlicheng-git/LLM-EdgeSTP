# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:04:35 2025

@author: HP
"""

from transformers import PretrainedConfig, AutoConfig

# 1. 定义模型配置类（存储所有超参数）
class LLM4STPConfig(PretrainedConfig):
    model_type = "llm4stp"  # 模型类型标识
    REGION_CONFIGS = {
        'caofeidian': {
            'lat_min': 38.72, 'lat_max': 39.10,
            'lon_min': 118.25, 'lon_max': 118.92,
            'cou_min': 0., 'cou_max': 359.9,
            'spd_min': 0., 'spd_max': 17.6
        },
        'osaka': {
            'lat_min': 33.72, 'lat_max': 34.18,
            'lon_min': 134.59, 'lon_max': 135.20,
            'cou_min': 0., 'cou_max': 359.9,
            'spd_min': 0., 'spd_max': 17.6
        }
    }

    def __init__(
        self,
        # 原始参数
        batch_size=16,
        llmmodel_hidden_size=896,
        out_dim_st=128,
        num_frequencies=10,
        max_time=100.,
        emd_use="True",
        periodicityencoding="True",
        kernel_size=3,
        in_channels=128,
        out_channels=128,
        seq_len=10,
        pred_len=10,
        pred_regress_len=2,
        num_vessels=2,
        # 新增参数
        filters=128,
        dropout=0.1,
        d_model=128, 
        area='osaka', 
        REGION_CONFIGS=REGION_CONFIGS,
        model_name="Qwen2",
        num_tokens=1000,  # 紧缩后的文本数量
        # MHAttentionLayer
        ffn_heads = 5,
        d_keys = 64,
        d_llm = None,
        # FlattenMultiHeads
        atten_num_heads=4,
        input_seq_len = 310,
        last_state_num = 256,
        # 优化参数
        learning_rate=0.0001,
        # region = REGION_CONFIGS[area],
        **kwargs
    ):
        self.llmmodel_hidden_size = llmmodel_hidden_size
        self.out_dim_st = out_dim_st
        self.num_frequencies = num_frequencies
        self.max_time = max_time
        self.emd_use = emd_use
        self.periodicityencoding = periodicityencoding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pred_regress_len = pred_regress_len
        self.num_vessels = num_vessels
        self.filters = filters
        self.dropout = dropout
        self.d_model = d_model
        self.region = REGION_CONFIGS[area]
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.ffn_heads = ffn_heads
        self.d_keys = d_keys
        self.d_llm = d_llm
        self.atten_num_heads = atten_num_heads
        self.input_seq_len = input_seq_len
        self.last_state_num = last_state_num
        # 优化参数
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        super().__init__(** kwargs)

LLM4STPConfig.register_for_auto_class()