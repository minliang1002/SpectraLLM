"""
季节性数据加载器 - 处理时间断点
确保滑动窗口不跨越时间断点
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAugmentor:
    """时间序列数据增强器"""
    
    def __init__(self, noise_std=0.05, scale_range=(0.9, 1.1), augment_prob=0.5):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.augment_prob = augment_prob
    
    def add_noise(self, x):
        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise
    
    def scale_amplitude(self, x):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale
    
    def jitter(self, x):
        jitter = np.random.uniform(-0.02, 0.02, x.shape)
        return x + jitter
    
    def augment(self, x, y, x_mark, y_mark):
        x_aug = x.copy()
        y_aug = y.copy()
        
        if np.random.random() < self.augment_prob:
            x_aug = self.add_noise(x_aug)
            y_aug = self.add_noise(y_aug)
        
        if np.random.random() < self.augment_prob:
            x_aug = self.scale_amplitude(x_aug)
            y_aug = self.scale_amplitude(y_aug)
        
        if np.random.random() < self.augment_prob:
            x_aug = self.jitter(x_aug)
            y_aug = self.jitter(y_aug)
        
        return x_aug, y_aug, x_mark, y_mark


class Dataset_Seasonal(Dataset):
    """
    季节性数据集加载器 - 处理时间断点
    """
    def __init__(self, root_path, flag='train', size=None,
                 data_path='train_spring.csv',
                 target_indices=[3, 4, 5, 6],
                 scale=True, stride=1,
                 augment=False, augment_factor=6,
                 noise_std=0.05, scale_range=(0.9, 1.1)):
        
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.stride = stride
        self.set_type = flag
        self.target_indices = target_indices
        self.scale = scale
        self.scaler = StandardScaler()
        self.root_path = root_path
        self.data_path = data_path
        
        self.augment = augment and (flag == 'train')
        self.augment_factor = augment_factor if self.augment else 1
        
        if self.augment:
            self.augmentor = TimeSeriesAugmentor(
                noise_std=noise_std,
                scale_range=scale_range
            )
        
        self.__read_data__()
    
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 创建datetime列用于检测断点
        df_raw['datetime'] = pd.to_datetime(df_raw[['Year', 'Month', 'Day', 'Hour']])
        df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
        
        # 添加时间编码
        df_raw['hour_sin'] = np.sin(2 * np.pi * df_raw['datetime'].dt.hour / 24).astype(np.float32)
        df_raw['hour_cos'] = np.cos(2 * np.pi * df_raw['datetime'].dt.hour / 24).astype(np.float32)
        
        # 检测时间断点
        time_diff = df_raw['datetime'].diff()
        gap_indices = time_diff[time_diff > pd.Timedelta(hours=1)].index.tolist()
        
        # 构建连续段
        segments = []
        start_idx = 0
        for gap_idx in gap_indices:
            if gap_idx > start_idx:
                segments.append((start_idx, gap_idx - 1))
            start_idx = gap_idx
        segments.append((start_idx, len(df_raw) - 1))
        
        print(f"  发现 {len(segments)} 个连续段")
        
        # 准备数据列
        cols_to_exclude = {'datetime', 'Year', 'Month', 'Day', 'Hour'}
        cols_data = [col for col in df_raw.columns if col not in cols_to_exclude and col not in ['hour_sin', 'hour_cos']]
        
        df_data = df_raw[cols_data].copy()
        df_data.insert(0, 'hour_cos', df_raw['hour_cos'])
        df_data.insert(0, 'hour_sin', df_raw['hour_sin'])
        
        # 标准化
        if self.scale:
            if self.set_type == 'train':
                self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                train_file = self.data_path.replace('val_', 'train_').replace('test_', 'train_')
                df_train = pd.read_csv(os.path.join(self.root_path, train_file))
                df_train['datetime'] = pd.to_datetime(df_train[['Year', 'Month', 'Day', 'Hour']])
                df_train['hour_sin'] = np.sin(2 * np.pi * df_train['datetime'].dt.hour / 24).astype(np.float32)
                df_train['hour_cos'] = np.cos(2 * np.pi * df_train['datetime'].dt.hour / 24).astype(np.float32)
                cols_data_train = [col for col in df_train.columns if col not in cols_to_exclude and col not in ['hour_sin', 'hour_cos']]
                df_data_train = df_train[cols_data_train].copy()
                df_data_train.insert(0, 'hour_cos', df_train['hour_cos'])
                df_data_train.insert(0, 'hour_sin', df_train['hour_sin'])
                
                self.scaler.fit(df_data_train.values)
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data
        self.data_y = data
        self.data_stamp = df_data[['hour_sin', 'hour_cos']].values.astype(np.float32)
        
        # 计算有效样本索引 - 只在连续段内生成样本
        self.valid_indices = []
        window_size = self.seq_len + self.pred_len
        
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len >= window_size:
                # 在这个段内生成样本
                num_samples = (seg_len - window_size) // self.stride + 1
                for i in range(num_samples):
                    idx = seg_start + i * self.stride
                    self.valid_indices.append(idx)
        
        self.base_len = len(self.valid_indices)
        
        aug_info = f", augment_factor={self.augment_factor}" if self.augment else ""
        print(f"{self.set_type} loaded: {len(self)} samples (stride={self.stride}{aug_info})")
        print(f"  有效样本数: {self.base_len}, 连续段数: {len(segments)}")
        print(f"  Target indices: {self.target_indices}")
    
    def __getitem__(self, index):
        base_index = index // self.augment_factor
        aug_index = index % self.augment_factor
        
        # 获取有效的起始位置
        s_begin = self.valid_indices[base_index]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end].copy()
        seq_y = self.data_y[r_begin:r_end].copy()
        seq_x_mark = self.data_stamp[s_begin:s_end].copy()
        seq_y_mark = self.data_stamp[r_begin:r_end].copy()
        
        if self.augment and aug_index > 0:
            seq_x, seq_y, seq_x_mark, seq_y_mark = self.augmentor.augment(
                seq_x, seq_y, seq_x_mark, seq_y_mark
            )
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return self.base_len * self.augment_factor
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
