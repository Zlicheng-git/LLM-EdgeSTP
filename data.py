# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:32:00 2025

@author: HP
"""

import numpy as np
import pandas as pd
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple#, List

class DataProcessor:
    def __init__(self, data_path: str, region: str = 'osaka', use_cols: Optional[list] = None):
        self.REGION_CONFIGS = {
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
        
        self.data_path = data_path
        self.region = region.lower()
        if self.region not in self.REGION_CONFIGS:
            raise ValueError(f"不支持的区域: {region}")

        self.config = self.REGION_CONFIGS[self.region]
        self.use_cols = use_cols 
        self.df = None

        # 数据存储
        self.train_enc_inputs = None
        self.test_enc_inputs = None
        self.train_dec_inputs = None
        self.test_dec_inputs = None
        self.train_y = None
        self.test_y = None

    def _min_max_normalize(self, df_chunk_values: np.ndarray) -> np.ndarray:
        chunk = df_chunk_values.copy()
        config = self.config
        chunk[:, 0] = (chunk[:, 0] - config['cou_min']) / (config['cou_max'] - config['cou_min'] + 1e-8)
        chunk[:, 1] = (chunk[:, 1] - config['spd_min']) / (config['spd_max'] - config['spd_min'] + 1e-8)
        chunk[:, 2] = (chunk[:, 2] - config['lon_min']) / (config['lon_max'] - config['lon_min'] + 1e-8)
        chunk[:, 3] = (chunk[:, 3] - config['lat_min']) / (config['lat_max'] - config['lat_min'] + 1e-8)
        return chunk

    def _generate_samples(self, input_len: int, output_len: int) -> Tuple[np.ndarray, np.ndarray]:
        print(f"生成样本: input_len={input_len}, output_len={output_len}...")

        mmsi_list = list(np.unique(self.df.MMSI_))
        inputs, outputs = [], []

        for mmsi in mmsi_list:
            vessel_data = self.df[self.df['MMSI_'] == mmsi].reset_index(drop=True)
            num_samples = len(vessel_data) - (input_len + output_len) + 1

            for j in range(num_samples):
                sample_input = vessel_data.iloc[j:j+input_len, 2:].values
                sample_output = vessel_data.iloc[j+input_len:j+input_len+output_len, 4:6].values
                inputs.append(sample_input)
                outputs.append(sample_output)

        X = np.array(inputs)
        y = np.array(outputs)
        print(f"生成 {X.shape[0]} 个样本.")
        return X, y

    def load_and_preprocess(self) -> 'DataProcessor':
        df = pd.read_csv(self.data_path, usecols=self.use_cols)
        df = df.loc[:, self.use_cols].copy()

        timefun_ = lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
        df['UnixTime_FEN'] = df['UnixTime_FEN'].apply(timefun_)
        df['MMSI_'] = df['MMSI_'].astype(int)
        df.fillna(value=0, inplace=True)

        for i in range((df.shape[1] - 2) // 4):
            start_col = 2 + 4 * i
            end_col = 6 + 4 * i
            raw_values = df.iloc[:, start_col:end_col].values
            normalized_values = self._min_max_normalize(raw_values)
            df.iloc[:, start_col:end_col] = normalized_values
        self.df = df
        return self

    def generate_samples(self, input_len: int, output_len: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._generate_samples(input_len, output_len)

    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> 'DataProcessor':
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = X[indices], y[indices]

        train_size = int(len(X_shuffled) * train_ratio)
        self.train_enc_inputs = X_shuffled[:train_size]
        self.test_enc_inputs  = X_shuffled[train_size:]
        self.train_dec_inputs = self.train_enc_inputs[:, :, 2:4]
        self.test_dec_inputs  = self.test_enc_inputs[:, :, 2:4]
        self.train_y = y_shuffled[:train_size]
        self.test_y  = y_shuffled[train_size:]

        print(f"训练集: {self.train_enc_inputs.shape[0]}, 测试集: {self.test_enc_inputs.shape[0]}")
        return self

    def process(self, input_len: int, output_len: int, train_ratio: float = 0.8) -> 'DataProcessor':
        self.load_and_preprocess()
        X, y = self.generate_samples(input_len, output_len)
        self.split_data(X, y, train_ratio)
        return self

    def get_all_data(self) -> Tuple[np.ndarray, ...]:
        if self.train_enc_inputs is None:
            raise RuntimeError("请先调用 .process() 方法.")
        return (
            self.train_enc_inputs,
            self.train_dec_inputs,
            self.train_y,
            self.test_enc_inputs,
            self.test_dec_inputs,
            self.test_y
        )
    
    def get_dataloaders(self, batch_size: int = 64, shuffle: bool = True):
        if self.train_enc_inputs is None:
            raise RuntimeError("请先调用 .process() 方法再获取数据加载器.")

        train_dataset = TensorDataset(
            torch.tensor(self.train_enc_inputs, dtype=torch.float32),
            torch.tensor(self.train_dec_inputs, dtype=torch.float32),
            torch.tensor(self.train_y, dtype=torch.float32).view(-1, 20)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.test_enc_inputs, dtype=torch.float32),
            torch.tensor(self.test_dec_inputs, dtype=torch.float32),
            torch.tensor(self.test_y, dtype=torch.float32).view(-1, 20)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def load_data(
        data_path: str,
        region: str = 'osaka',
        input_len: int = 10,
        output_len: int = 20,
        train_ratio: float = 0.8,
        use_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, ...]:
        print("="*60)
        print(f"区域: {region}, 输入长度: {input_len}, 输出长度: {output_len}, 训练集比例: {train_ratio}")
        print("="*60)

        processor = DataProcessor(
            data_path=data_path,
            region=region,
            use_cols=use_cols
        )
        processor.process(input_len=input_len, output_len=output_len, train_ratio=train_ratio)

        return processor.get_all_data()