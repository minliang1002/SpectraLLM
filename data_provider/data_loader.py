import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import os
import torch

class Dataset_ASU_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='ASUh.csv',
                 target='Tamp_E', scale=False, timeenc=0, freq='h', 
                 seasonal_patterns=None, percent=100): 
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = StandardScaler()  
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['datetime'] = pd.to_datetime(df_raw[['Year', 'Month', 'Day', 'Hour']])
        
        data_x, data_y, data_stamp = [], [], []
        
        # Define the target period for the test and validation sets
        target_datetime_start = pd.Timestamp(year=2019, month=12, day=1)
        target_datetime_end = pd.Timestamp(year=2019, month=12, day=31, hour=23)

        # Split the dataset
        train_df = df_raw[df_raw['datetime'] < target_datetime_start]
        test_val_df = df_raw[(df_raw['datetime'] >= target_datetime_start) & (df_raw['datetime'] <= target_datetime_end)]
        # mid_point = len(test_val_df) // 2
        # val_df = test_val_df.iloc[:mid_point]
        # test_df = test_val_df.iloc[mid_point:]
        val_df = test_val_df[:]
        test_df = val_df

        # Select dataset based on set_type
        if self.set_type == 0:
            selected_df = train_df
        elif self.set_type == 1:
            selected_df = val_df
        else:
            selected_df = test_df

        # Calculate time encoding for the selected dataset
        selected_df['hour_sin'] = np.sin(2 * np.pi * selected_df['datetime'].dt.hour / 24).astype(np.float32)
        selected_df['hour_cos'] = np.cos(2 * np.pi * selected_df['datetime'].dt.hour / 24).astype(np.float32)

        # Exclude datetime, Year, Month, Day, and Hour from features while maintaining order
        cols_to_exclude = {'datetime', 'Year', 'Month', 'Day', 'Hour'}
        cols_data = [col for col in df_raw.columns if col not in cols_to_exclude]
        df_data = selected_df[cols_data]

        # Insert hour_sin and hour_cos at the beginning of df_data
        df_data.insert(0, 'hour_cos', selected_df['hour_cos'])
        df_data.insert(0, 'hour_sin', selected_df['hour_sin'])


        # Data normalization
        if self.scale:
            self.scaler.fit(train_df[cols_data].values)  # Fit scaler on numeric columns only
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Special handling for 'MS' features
        if self.features == 'MS':
            output_columns = selected_df.columns[5:9]  # Adjusted output columns
            df_output = selected_df.loc[:, output_columns]
            if self.scale:
                self.scaler.fit(df_output.values)
                output_data = self.scaler.transform(df_output.values)
            else:
                output_data = df_output.values
            data_y.append(output_data)
        else:
            data_y.append(data)

        data_x.append(data)
        data_stamp.append(df_data[['hour_sin', 'hour_cos']].values.astype(np.float32))
        
        self.data_x = np.concatenate(data_x, axis=0)
        self.data_y = np.concatenate(data_y, axis=0)
        self.data_stamp = np.concatenate(data_stamp, axis=0)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Ensure data is tensor
        seq_x = torch.tensor(self.data_x[s_begin:s_end].astype(np.float32))  # Ensure data is float32
        seq_y = torch.tensor(self.data_y[r_begin:r_end].astype(np.float32))  # Ensure data is float32
        seq_x_mark = torch.tensor(self.data_stamp[s_begin:s_end].astype(np.float32))  # Ensure data is float32
        seq_y_mark = torch.tensor(self.data_stamp[r_begin:r_end].astype(np.float32))  # Ensure data is float32

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)