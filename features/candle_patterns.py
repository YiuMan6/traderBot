import numpy as np
import pandas as pd

def identify_patterns(df):
    """识别K线形态"""
    # 确保数据包含必要的列
    required_cols = ['open', 'high', 'low', 'close','time']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("数据缺少必要的列：open, high, low, close")
    
    # 复制数据框以避免修改原始数据
    df = df.copy()
    
    # 计算实体和影线
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_size'] = abs(df['body'])
    
    # 1. 锤子线
    df['hammer'] = (
        (df['lower_shadow'] > 2 * df['body_size']) &  # 下影线长度是实体的2倍以上
        (df['upper_shadow'] < df['body_size']) &      # 上影线较短
        (df['body_size'] > 0)                         # 实体存在
    ).astype(int)
    
    # 2. 十字星
    df['doji'] = (
        (abs(df['body']) < (df['high'] - df['low']) * 0.1) &  # 实体很小
        (df['high'] > df[['open', 'close']].max(axis=1)) &    # 有上下影线
        (df['low'] < df[['open', 'close']].min(axis=1))
    ).astype(int)
    
    # 3. 吞没形态
    df['engulfing'] = 0
    for i in range(1, len(df)):
        if (df['body'].iloc[i-1] < 0 and df['body'].iloc[i] > 0 and  # 前阴后阳
            abs(df['body'].iloc[i]) > abs(df['body'].iloc[i-1])):    # 后面的实体吞没前面的
            df.loc[df.index[i], 'engulfing'] = 1
    
    # 4. 启明星
    df['morning_star'] = 0
    for i in range(2, len(df)):
        if (df['body'].iloc[i-2] < 0 and                     # 第一天阴线
            abs(df['body'].iloc[i-1]) < df['body_size'].mean() * 0.5 and  # 第二天十字星
            df['body'].iloc[i] > 0 and                       # 第三天阳线
            df['close'].iloc[i] > df['open'].iloc[i-2]):     # 收盘价高于第一天开盘价
            df.loc[df.index[i], 'morning_star'] = 1
    
    # 5. 上吊线
    df['hanging_man'] = (
        (df['lower_shadow'] > 2 * df['body_size']) &
        (df['upper_shadow'] < df['body_size']) &
        (df['close'] < df['open'])
    ).astype(int)
    
    # 添加 _pattern 后缀
    pattern_columns = ['hammer', 'doji', 'engulfing', 'morning_star', 'hanging_man']
    for col in pattern_columns:
        df[f'{col}_pattern'] = df[col]
        df.drop(col, axis=1, inplace=True)
    
    # 删除临时列
    df.drop(['body', 'upper_shadow', 'lower_shadow', 'body_size'], axis=1, inplace=True)
    
    return df