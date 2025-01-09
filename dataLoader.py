import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import pandas_ta as ta  
import openpyxl

from datetime import datetime
from features.candle_patterns import identify_patterns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(x, y, lookback):
    sequences, labels = [], []
    for i in range(len(x) - lookback):
        sequences.append(x[i:(i + lookback)])
        labels.append(y[i + lookback])
    return np.array(sequences), np.array(labels)

def load_data(file_path, batch_size, lookback=30):
    print(f"\n正在加载数据: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 转换为DataFrame并确保使用复制
    df = pd.DataFrame(data).copy()
    
    # 时间处理
    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 添加技术指标
    ## k 线形态指标
    df = identify_patterns(df)
    
    # 删除包含NaN的行
    df = df.dropna().reset_index(drop=True)
    
    # 使用 loc 添加新列
    df.loc[:, 'future_return'] = (df['close'].shift(-1) / df['close'] - 1) * 100
    df.loc[:, 'future_direction'] = (df['future_return'] > 0).astype(int)
    
    # 准备特征
    feature_columns = [col for col in df.columns if col not in ['time', 'future_return', 'future_direction']]
    
    X = df[feature_columns].values
    y = df['future_direction'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建序列
    X_seq, y_seq = create_sequences(X_scaled, y, lookback)
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_seq, y_seq)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False,
        num_workers=4,
        persistent_workers=True
    )

    return {
        'train_loader': train_loader,
        'scaler': scaler,
        'feature_dim': len(feature_columns),
        'lookback': lookback,
        'train_size': len(train_dataset),
        'time_range': (df['time'].min(), df['time'].max()),
        'feature_names': feature_columns,
        'df': df  # 添加原始DataFrame以便后续分析
    }
  