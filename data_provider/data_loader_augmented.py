"""
带数据增强的多目标数据加载器
支持多种时间序列数据增强策略
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAugmentor:
    """时间序列数据增强器"""
    
    def __init__(self, 
                 noise_std=0.05,        # 噪声标准差
                 scale_range=(0.9, 1.1), # 幅度缩放范围
                 time_warp_sigma=0.2,    # 时间扭曲强度
                 mixup_alpha=0.2,        # Mixup参数
                 augment_prob=0.5):      # 每种增强的应用概率
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_sigma = time_warp_sigma
        self.mixup_alpha = mixup_alpha
        self.augment_prob = augment_prob
    
    def add_noise(self, x):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise
    
    def scale_amplitude(self, x):
        """随机缩放幅度"""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale
    
    def time_warp(self, x):
        """时间扭曲 - 使用三次样条插值"""
        seq_len = x.shape[0]
        if seq_len < 4:
            return x
        
        try:
            # 生成扭曲的时间点
            orig_steps = np.arange(seq_len)
            random_warps = np.random.normal(loc=1.0, scale=self.time_warp_sigma, size=(4,))
            # 确保所有值为正
            random_warps = np.abs(random_warps) + 0.1
            warp_steps = np.linspace(0, seq_len-1, num=4)
            warped_steps = np.cumsum(random_warps)
            warped_steps = (warped_steps - warped_steps[0]) / (warped_steps[-1] - warped_steps[0]) * (seq_len - 1)
            
            # 确保严格单调递增
            for i in range(1, len(warped_steps)):
                if warped_steps[i] <= warped_steps[i-1]:
                    warped_steps[i] = warped_steps[i-1] + 0.01
            
            # 对每个特征进行插值
            warped_x = np.zeros_like(x)
            for i in range(x.shape[1]):
                cs = CubicSpline(warped_steps, x[warp_steps.astype(int), i])
                warped_x[:, i] = cs(orig_steps)
            
            return warped_x
        except Exception:
            # 如果插值失败，返回原始数据
            return x
    
    def jitter(self, x):
        """抖动 - 添加小幅度随机偏移"""
        jitter = np.random.uniform(-0.02, 0.02, x.shape)
        return x + jitter
    
    def window_slice(self, x, y, x_mark, y_mark, slice_ratio=0.9):
        """窗口切片 - 随机选择子窗口"""
        seq_len = x.shape[0]
        target_len = int(seq_len * slice_ratio)
        if target_len < seq_len:
            start = np.random.randint(0, seq_len - target_len + 1)
            # 需要padding回原长度
            sliced_x = x[start:start+target_len]
            # 使用插值恢复原长度
            indices = np.linspace(0, target_len-1, seq_len)
            new_x = np.zeros_like(x)
            for i in range(x.shape[1]):
                new_x[:, i] = np.interp(np.arange(seq_len), 
                                        np.linspace(0, seq_len-1, target_len), 
                                        sliced_x[:, i])
            return new_x
        return x
    
    def augment(self, x, y, x_mark, y_mark):
        """应用随机数据增强组合"""
        x_aug = x.copy()
        y_aug = y.copy()
        
        # 随机应用各种增强
        if np.random.random() < self.augment_prob:
            x_aug = self.add_noise(x_aug)
            y_aug = self.add_noise(y_aug)
        
        if np.random.random() < self.augment_prob:
            x_aug = self.scale_amplitude(x_aug)
            y_aug = self.scale_amplitude(y_aug)
        
        if np.random.random() < self.augment_prob * 0.5:  # 时间扭曲概率降低
            x_aug = self.time_warp(x_aug)
        
        if np.random.random() < self.augment_prob:
            x_aug = self.jitter(x_aug)
            y_aug = self.jitter(y_aug)
        
        return x_aug, y_aug, x_mark, y_mark


class Dataset_Multi_Target_Augmented(Dataset):
    """
    带数据增强的多目标数据集加载器
    
    Args:
        augment: 是否启用数据增强（仅训练集有效）
        augment_factor: 数据增强倍数，例如4表示每个原始样本生成4个增强样本
        noise_std: 噪声标准差
        scale_range: 幅度缩放范围
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train_spring.csv',
                 target_indices=[3, 4, 5, 6],
                 scale=True, timeenc=0, freq='h', 
                 percent=100, seasonal_patterns=None,
                 stride=1,
                 # 数据增强参数
                 augment=False,
                 augment_factor=4,
                 noise_std=0.05,
                 scale_range=(0.9, 1.1)):
        
        if size is None:
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.stride = stride
        
        assert flag in ['train', 'test', 'val']
        self.set_type = flag
        
        self.features = features
        self.target_indices = target_indices
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = StandardScaler()
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        
        # 数据增强设置（仅训练集）
        self.augment = augment and (flag == 'train')
        self.augment_factor = augment_factor if self.augment else 1
        
        if self.augment:
            self.augmentor = TimeSeriesAugmentor(
                noise_std=noise_std,
                scale_range=scale_range
            )
        
        self.__read_data__()
    
    def __read_data__(self):
        """读取数据"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        df_raw['datetime'] = pd.to_datetime(df_raw[['Year', 'Month', 'Day', 'Hour']])
        df_raw['hour_sin'] = np.sin(2 * np.pi * df_raw['datetime'].dt.hour / 24).astype(np.float32)
        df_raw['hour_cos'] = np.cos(2 * np.pi * df_raw['datetime'].dt.hour / 24).astype(np.float32)
        
        cols_to_exclude = {'datetime', 'Year', 'Month', 'Day', 'Hour'}
        cols_data = [col for col in df_raw.columns if col not in cols_to_exclude and col not in ['hour_sin', 'hour_cos']]
        
        df_data = df_raw[cols_data].copy()
        df_data.insert(0, 'hour_cos', df_raw['hour_cos'])
        df_data.insert(0, 'hour_sin', df_raw['hour_sin'])
        
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
        
        # 计算原始样本数
        total_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.base_len = (total_len - 1) // self.stride + 1
        
        aug_info = f", augment_factor={self.augment_factor}" if self.augment else ""
        print(f"{self.set_type} loaded: {len(self)} samples (stride={self.stride}{aug_info})")
        print(f"  Target indices: {self.target_indices}")
    
    def __getitem__(self, index):
        """获取样本，支持数据增强"""
        # 计算原始样本索引和增强索引
        base_index = index // self.augment_factor
        aug_index = index % self.augment_factor
        
        # 获取原始数据
        s_begin = base_index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end].copy()
        seq_y = self.data_y[r_begin:r_end].copy()
        seq_x_mark = self.data_stamp[s_begin:s_end].copy()
        seq_y_mark = self.data_stamp[r_begin:r_end].copy()
        
        # 如果是增强样本（aug_index > 0），应用数据增强
        if self.augment and aug_index > 0:
            seq_x, seq_y, seq_x_mark, seq_y_mark = self.augmentor.augment(
                seq_x, seq_y, seq_x_mark, seq_y_mark
            )
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return self.base_len * self.augment_factor
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
