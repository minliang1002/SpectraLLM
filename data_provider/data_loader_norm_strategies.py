"""
不同归一化策略的数据加载器
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class Dataset_ASU_Seasonal_NormStrategy(Dataset):
    """
    支持多种归一化策略的季节性数据集加载器
    
    归一化策略:
    1. 'standard': StandardScaler (当前使用)
    2. 'minmax': MinMaxScaler (0-1归一化)
    3. 'robust': RobustScaler (对异常值鲁棒)
    4. 'target_only': 只归一化目标列
    5. 'per_feature': 每个特征独立归一化
    6. 'log_transform': 对目标列做log变换后归一化
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='train_spring.csv',
                 target='Tamp_E', scale=True, timeenc=0, freq='h', 
                 percent=100, seasonal_patterns=None,
                 norm_strategy='standard'):
        
        if size is None:
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        self.set_type = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.norm_strategy = norm_strategy
        
        # 根据策略选择scaler
        if norm_strategy == 'minmax':
            self.scaler = MinMaxScaler()
        elif norm_strategy == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.target_scaler = StandardScaler()  # 用于target_only策略
        
        self.__read_data__()
    
    def __read_data__(self):
        """读取数据"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 创建datetime列
        df_raw['datetime'] = pd.to_datetime(df_raw[['Year', 'Month', 'Day', 'Hour']])
        
        # 计算时间编码
        df_raw['hour_sin'] = np.sin(2 * np.pi * df_raw['datetime'].dt.hour / 24).astype(np.float32)
        df_raw['hour_cos'] = np.cos(2 * np.pi * df_raw['datetime'].dt.hour / 24).astype(np.float32)
        
        # 排除datetime和时间列
        cols_to_exclude = {'datetime', 'Year', 'Month', 'Day', 'Hour'}
        cols_data = [col for col in df_raw.columns if col not in cols_to_exclude and col not in ['hour_sin', 'hour_cos']]
        
        # 构建特征数据
        df_data = df_raw[cols_data].copy()
        
        # 在开头插入时间编码
        df_data.insert(0, 'hour_cos', df_raw['hour_cos'])
        df_data.insert(0, 'hour_sin', df_raw['hour_sin'])
        
        # 获取目标列索引
        target_idx = cols_data.index(self.target)
        self.target_col_idx = 2 + target_idx  # +2是因为前面有hour_sin和hour_cos
        
        # 数据归一化
        if self.scale:
            data = self._apply_normalization(df_data, cols_data, cols_to_exclude)
        else:
            data = df_data.values
        
        # 处理MS特征
        if self.features == 'MS':
            output_data = data[:, self.target_col_idx:self.target_col_idx + 1]
        else:
            output_data = data
        
        self.data_x = data
        self.data_y = output_data
        self.data_stamp = df_data[['hour_sin', 'hour_cos']].values.astype(np.float32)
        
        print(f"{self.set_type} loaded: {len(self.data_x)} samples (norm: {self.norm_strategy})")
    
    def _apply_normalization(self, df_data, cols_data, cols_to_exclude):
        """应用不同的归一化策略"""
        
        if self.norm_strategy == 'target_only':
            # 策略4: 只归一化目标列
            return self._normalize_target_only(df_data, cols_data, cols_to_exclude)
        
        elif self.norm_strategy == 'per_feature':
            # 策略5: 每个特征独立归一化（不共享scaler）
            return self._normalize_per_feature(df_data)
        
        elif self.norm_strategy == 'log_transform':
            # 策略6: 对目标列做log变换
            return self._normalize_with_log(df_data, cols_data, cols_to_exclude)
        
        else:
            # 策略1-3: standard/minmax/robust
            return self._normalize_standard(df_data, cols_to_exclude)
    
    def _normalize_standard(self, df_data, cols_to_exclude):
        """标准归一化（当前方法）"""
        if self.set_type == 'train':
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            # 加载训练集来fit scaler
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
        
        return data
    
    def _normalize_target_only(self, df_data, cols_data, cols_to_exclude):
        """只归一化目标列，其他特征保持原始尺度"""
        data = df_data.values.copy()
        
        if self.set_type == 'train':
            # 只对目标列fit scaler
            target_data = data[:, self.target_col_idx:self.target_col_idx+1]
            self.target_scaler.fit(target_data)
            data[:, self.target_col_idx:self.target_col_idx+1] = self.target_scaler.transform(target_data)
        else:
            # 加载训练集的目标列来fit
            train_file = self.data_path.replace('val_', 'train_').replace('test_', 'train_')
            df_train = pd.read_csv(os.path.join(self.root_path, train_file))
            train_target = df_train[self.target].values.reshape(-1, 1)
            self.target_scaler.fit(train_target)
            
            target_data = data[:, self.target_col_idx:self.target_col_idx+1]
            data[:, self.target_col_idx:self.target_col_idx+1] = self.target_scaler.transform(target_data)
        
        # 保存完整的scaler用于反归一化
        self.scaler = self.target_scaler
        
        return data
    
    def _normalize_per_feature(self, df_data):
        """每个特征独立归一化"""
        data = df_data.values.copy()
        
        # 为每个特征创建独立的scaler
        if not hasattr(self, 'feature_scalers'):
            self.feature_scalers = {}
        
        if self.set_type == 'train':
            for i in range(data.shape[1]):
                scaler = StandardScaler()
                data[:, i:i+1] = scaler.fit_transform(data[:, i:i+1])
                self.feature_scalers[i] = scaler
        else:
            # 加载训练集来fit每个特征的scaler
            train_file = self.data_path.replace('val_', 'train_').replace('test_', 'train_')
            df_train = pd.read_csv(os.path.join(self.root_path, train_file))
            df_train['datetime'] = pd.to_datetime(df_train[['Year', 'Month', 'Day', 'Hour']])
            df_train['hour_sin'] = np.sin(2 * np.pi * df_train['datetime'].dt.hour / 24).astype(np.float32)
            df_train['hour_cos'] = np.cos(2 * np.pi * df_train['datetime'].dt.hour / 24).astype(np.float32)
            
            # 重建训练数据
            cols_to_exclude = {'datetime', 'Year', 'Month', 'Day', 'Hour'}
            cols_data_train = [col for col in df_train.columns if col not in cols_to_exclude and col not in ['hour_sin', 'hour_cos']]
            df_data_train = df_train[cols_data_train].copy()
            df_data_train.insert(0, 'hour_cos', df_train['hour_cos'])
            df_data_train.insert(0, 'hour_sin', df_train['hour_sin'])
            train_data = df_data_train.values
            
            for i in range(data.shape[1]):
                scaler = StandardScaler()
                scaler.fit(train_data[:, i:i+1])
                data[:, i:i+1] = scaler.transform(data[:, i:i+1])
                self.feature_scalers[i] = scaler
        
        # 使用目标列的scaler作为主scaler
        self.scaler = self.feature_scalers[self.target_col_idx]
        
        return data
    
    def _normalize_with_log(self, df_data, cols_data, cols_to_exclude):
        """对目标列做log变换后归一化"""
        data = df_data.values.copy()
        
        # 对目标列做log变换
        target_data = data[:, self.target_col_idx]
        # 确保所有值都是正数
        min_val = target_data.min()
        if min_val <= 0:
            target_data = target_data - min_val + 1
        
        log_target = np.log1p(target_data).reshape(-1, 1)
        
        if self.set_type == 'train':
            self.target_scaler.fit(log_target)
            data[:, self.target_col_idx:self.target_col_idx+1] = self.target_scaler.transform(log_target)
            self.log_offset = min_val if min_val <= 0 else 0
        else:
            # 加载训练集
            train_file = self.data_path.replace('val_', 'train_').replace('test_', 'train_')
            df_train = pd.read_csv(os.path.join(self.root_path, train_file))
            train_target = df_train[self.target].values
            
            min_val_train = train_target.min()
            if min_val_train <= 0:
                train_target = train_target - min_val_train + 1
            log_train_target = np.log1p(train_target).reshape(-1, 1)
            
            self.target_scaler.fit(log_train_target)
            data[:, self.target_col_idx:self.target_col_idx+1] = self.target_scaler.transform(log_target)
            self.log_offset = min_val_train if min_val_train <= 0 else 0
        
        self.scaler = self.target_scaler
        
        return data
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """反归一化"""
        if self.norm_strategy == 'log_transform':
            # 先反归一化，再反log变换
            denorm = self.scaler.inverse_transform(data)
            original = np.expm1(denorm)
            if hasattr(self, 'log_offset') and self.log_offset != 0:
                original = original + self.log_offset - 1
            return original
        else:
            return self.scaler.inverse_transform(data)
