from data_provider.data_loader import  Dataset_ASU_hour
from data_provider.data_loader_seasonal import Dataset_Seasonal as Dataset_ASU_Seasonal
from data_provider.data_loader_norm_strategies import Dataset_ASU_Seasonal_NormStrategy
from data_provider.data_loader_multi_target import Dataset_Multi_Target
from data_provider.data_loader_augmented import Dataset_Multi_Target_Augmented
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import StandardScaler
import pickle  # 用于保存和加载scaler
import pandas as pd
data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    # 'nps': NPSloader,
    # 'nps_custom': Dataset_NPS_Custom,
    'ASU': Dataset_ASU_hour,
    'ASU_Seasonal': Dataset_ASU_Seasonal,  # 季节性划分使用专门的数据加载器
    'ASU_Seasonal_Norm': Dataset_ASU_Seasonal_NormStrategy,  # 支持多种归一化策略
    'Multi_Target': Dataset_Multi_Target,  # 多目标数据加载器
    'Multi_Target_Aug': Dataset_Multi_Target_Augmented  # 带数据增强的多目标数据加载器
}


def load_all_train_data(root_path, data_path):
    df_train = pd.read_csv(os.path.join(root_path, data_path))
    return df_train

    

def data_provider(args, flag, is_shuffle=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    
    # 使用args.data_path作为数据文件名
    data_path = args.data_path

    if flag == 'test':
        if is_shuffle is None:
            shuffle_flag = False
        else:
            shuffle_flag = is_shuffle
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        if is_shuffle is None:
            shuffle_flag = True
        else:
            shuffle_flag = is_shuffle
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        
        # 构建数据集参数
        dataset_kwargs = {
            'root_path': args.root_path,
            'data_path': data_path,
            'flag': flag,
            'size': [args.seq_len, args.label_len, args.pred_len],
            'features': args.features,
            'target': args.target,
            'timeenc': 0,
            'percent': percent,
            'freq': freq,
            'seasonal_patterns': args.seasonal_patterns,
        }
        
        # 如果使用归一化策略数据集，添加norm_strategy参数
        if args.data == 'ASU_Seasonal_Norm' and hasattr(args, 'norm_strategy'):
            dataset_kwargs['norm_strategy'] = args.norm_strategy
        
        # 如果使用ASU_Seasonal数据集，使用target_indices而不是features/target
        if args.data in ['ASU_Seasonal', 'ASU_Seasonal_Norm'] and hasattr(args, 'target_indices'):
            dataset_kwargs['target_indices'] = args.target_indices
            # ASU_Seasonal只需要: root_path, data_path, flag, size, target_indices
            # 删除不需要的参数
            for key in ['features', 'target', 'timeenc', 'seasonal_patterns', 'percent', 'freq']:
                if key in dataset_kwargs:
                    del dataset_kwargs[key]
        
        # 如果使用多目标数据集，添加target_indices参数
        if args.data in ['Multi_Target', 'Multi_Target_Aug'] and hasattr(args, 'target_indices'):
            dataset_kwargs['target_indices'] = args.target_indices
            # 多目标不需要单个target参数
            del dataset_kwargs['target']
            # 添加stride参数
            if hasattr(args, 'stride'):
                dataset_kwargs['stride'] = args.stride
            
            # 如果使用增强数据集，添加增强参数
            if args.data == 'Multi_Target_Aug':
                if hasattr(args, 'augment'):
                    dataset_kwargs['augment'] = args.augment
                if hasattr(args, 'augment_factor'):
                    dataset_kwargs['augment_factor'] = args.augment_factor
                if hasattr(args, 'noise_std'):
                    dataset_kwargs['noise_std'] = args.noise_std
        
        data_set = Data(**dataset_kwargs)
        batch_size = args.batch_size
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
