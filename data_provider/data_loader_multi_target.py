"""
多目标数据加载器
专门为Multi-Task模型设计，返回所有目标特征的未来值
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class Dataset_Multi_Target(Dataset):
    """
    多目标数据集加载器
    返回所有特征的历史值和指定目标特征的未来值
    
    Args:
        stride: 窗口滑动步长，默认为1。较大的步长可以减少相邻窗口的重叠，
                避免FFT特征过度相似导致的过平滑问题。
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train_spring.csv',
                 target_indices=[3, 4, 5, 6],  # Tamp_E, Tamp_C, Tamp_H, Tamp_Comb
                 scale=True, timeenc=0, freq='h', 
                 percent=100, seasonal_patterns=None,
                 stride=1):  # 新增stride参数
        
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.stride = stride  # 窗口滑动步长
        
        # 根据flag确定文件名
        assert flag in ['train', 'test', 'val']
        self.set_type = flag
        
        self.features = features
        self.target_indices = target_indices  # 多个目标索引
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = StandardScaler()
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        """读取数据"""
        # 根据flag确定实际的数据文件
        if self.set_type == 'train':
            actual_file = self.data_path
        elif self.set_type == 'val':
            actual_file = self.data_path.replace('train_', 'val_')
        else:  # test
            actual_file = self.data_path.replace('train_', 'test_')
        
        df_raw = pd.read_csv(os.path.join(self.root_path, actual_file))
        print(f"Loading {self.set_type} data from: {actual_file}")
        
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
        
        # 数据归一化
        if self.scale:
            # 对于训练集，fit scaler
            # 对于验证集和测试集，使用训练集的scaler
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
                
                # 用训练集fit，然后transform当前数据
                self.scaler.fit(df_data_train.values)
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 保存完整数据 (用于输入和输出)
        self.data_x = data
        self.data_y = data  # 多目标：输出也是完整数据
        self.data_stamp = df_data[['hour_sin', 'hour_cos']].values.astype(np.float32)
        
        print(f"{self.set_type} loaded: {len(self)} samples (stride={self.stride})")
        print(f"  Target indices: {self.target_indices}")
    
    def __getitem__(self, index):
        """
        返回:
            seq_x: [seq_len, n_features] 历史数据
            seq_y: [pred_len, n_features] 未来数据 (包含所有特征)
            seq_x_mark: [seq_len, 2] 历史时间编码
            seq_y_mark: [pred_len, 2] 未来时间编码
        """
        # 使用stride计算实际的数据起始位置
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end  # 未来数据从seq_len之后开始
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]  # [seq_len, n_features]
        seq_y = self.data_y[r_begin:r_end]  # [pred_len, n_features] 包含所有特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 根据stride计算样本数量
        total_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return (total_len - 1) // self.stride + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
