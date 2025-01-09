import pandas_ta as ta  
import pandas as pd


def identify_patterns(df):
    # print(df,'df')
    """添加技术分析特征"""
    # print("\n=== 开始添加技术指标 ===")
    # print(f"初始特征数: {len(df.columns)}")
    # print(f"初始列名: {df.columns.tolist()}")
    # df = pd.DataFrame()

# Help about this, 'ta', extension
    # help(df.ta)

# List of all indicators
    # df.ta.indicators()

# Help about an indicator such as bbands
    # help(ta.bbands)

    # 1. K线形态特征
    print("\n1. 添加K线形态特征...")
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
    
    # 添加锥子线特征
    df['body'] = abs(df['close'] - df['open'])
    df['prev_body'] = df['body'].shift(1)
    df['upper_shadow_abs'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow_abs'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # 计算锥子线条件
    df['is_spindle'] = (
        (df['body'] > 0.6 * (df['upper_shadow_abs'] + df['lower_shadow_abs'])) &  # 实体占比大
        (df['body'] > df['prev_body'] * 1.5) &                                    # 实体比前一根大
        (df['upper_shadow_abs'] < df['body'] * 0.3) &                            # 上影线短
        (df['lower_shadow_abs'] < df['body'] * 0.3)                              # 下影线短
    )
    
    # 添加方向
    df['spindle'] = 0
    df.loc[(df['is_spindle']) & (df['close'] > df['open']), 'spindle'] = 1    # 看涨锥子
    df.loc[(df['is_spindle']) & (df['close'] < df['open']), 'spindle'] = -1   # 看跌锥子
    
    # 删除中间计算列
    df = df.drop(['body', 'prev_body', 'upper_shadow_abs', 'lower_shadow_abs', 'is_spindle'], axis=1)
    
    print(f"添加K线特征后的特征数: {len(df.columns)}")
    
    # 2. 技术指标
    print("\n2. 添加技术指标...")
    # 使用pandas_ta添加技术指标
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    print(f"添加技术指标后的特征数: {len(df.columns)}")
    
    # 3. 成交量特征
    print("\n3. 添加成交量特征...")
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    print(f"添加成交量特征后的特征数: {len(df.columns)}")
    
    # 4. 价格与均线关系
    print("\n4. 添加价格均线关系...")
    df['price_ma5_ratio'] = df['close'] / df['SMA_5']
    df['price_ma10_ratio'] = df['close'] / df['SMA_10']
    df['price_ma20_ratio'] = df['close'] / df['SMA_20']
    print(f"添加价格均线关系后的特征数: {len(df.columns)}")
    
    # 打印最终特征列表
    print("\n=== 最终特征列表 ===")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    return df