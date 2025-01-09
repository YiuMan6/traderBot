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
    
    # 计算基础特征
    df['body'] = abs(df['close'] - df['open'])
    df['body_prev'] = df['body'].shift(1)
    df['body_prev2'] = df['body'].shift(2)
    df['close_prev'] = df['close'].shift(1)
    df['open_prev'] = df['open'].shift(1)
    df['close_prev2'] = df['close'].shift(2)
    df['open_prev2'] = df['open'].shift(2)
    
    # 计算实体中点
    df['mid_point'] = (df['open'] + df['close']) / 2
    df['mid_point_prev'] = (df['open_prev'] + df['close_prev']) / 2
    df['mid_point_prev2'] = (df['open_prev2'] + df['close_prev2']) / 2
    
    # 流星线形态 (看跌反转)
    df['shooting_star'] = 0
    shooting_star_condition = (
        (df['upper_shadow'] > 2 * df['body_ratio']) &     # 上影线至少是实体的2倍
        (df['lower_shadow'] < 0.1) &                      # 几乎没有下影线
        (df['body'] < df['body_prev'] * 0.5) &           # 实体小于前一天的一半
        (df['close'] < df['close_prev'])                  # 收盘价低于前一天
    )
    df.loc[shooting_star_condition, 'shooting_star'] = 1
    
    # 倒锤子形态 (看涨反转)
    df['inverted_hammer'] = 0
    inverted_hammer_condition = (
        (df['upper_shadow'] > 2 * df['body_ratio']) &     # 上影线至少是实体的2倍
        (df['lower_shadow'] < 0.1) &                      # 几乎没有下影线
        (df['body'] < df['body_prev'] * 0.5) &           # 实体小于前一天的一半
        (df['close'] > df['close_prev'])                  # 收盘价高于前一天
    )
    df.loc[inverted_hammer_condition, 'inverted_hammer'] = 1
    
    # 启明星模式 (看涨反转)
    df['morning_star'] = 0
    morning_star_condition = (
        (df['close_prev2'] < df['open_prev2']) &                    # 第一天是阴线
        (df['body_prev2'] > df['body_prev'] * 2) &                 # 第一天实体大于第二天
        (df['mid_point_prev'] < df['close_prev2']) &               # 第二天位于第一天下方
        (df['close'] > df['open']) &                               # 第三天是阳线
        (df['close'] > df['mid_point_prev2'])                      # 第三天收盘价高于第一天中点
    )
    df.loc[morning_star_condition, 'morning_star'] = 1
    
    # 黄昏星模式 (看跌反转)
    df['evening_star'] = 0
    evening_star_condition = (
        (df['close_prev2'] > df['open_prev2']) &                    # 第一天是阳线
        (df['body_prev2'] > df['body_prev'] * 2) &                 # 第一天实体大于第二天
        (df['mid_point_prev'] > df['close_prev2']) &               # 第二天位于第一天上方
        (df['close'] < df['open']) &                               # 第三天是阴线
        (df['close'] < df['mid_point_prev2'])                      # 第三天收盘价低于第一天中点
    )
    df.loc[evening_star_condition, 'evening_star'] = 1
    
    # 添加孕线形态检测
    # 看涨孕线：第一天大阴线，第二天小阳线被包含在第一天实体内
    df['bullish_harami'] = 0
    bullish_harami_condition = (
        (df['close_prev'] < df['open_prev']) &                    # 第一天是阴线
        (df['close'] > df['open']) &                             # 第二天是阳线
        (df['body'] < df['body_prev'] * 0.5) &                   # 第二天实体小于第一天的一半
        (df['open'] > df['close_prev']) &                        # 第二天开盘价高于第一天收盘价
        (df['close'] < df['open_prev'])                          # 第二天收盘价低于第一天开盘价
    )
    df.loc[bullish_harami_condition, 'bullish_harami'] = 1
    
    # 看跌孕线：第一天大阳线，第二天小阴线被包含在第一天实体内
    df['bearish_harami'] = 0
    bearish_harami_condition = (
        (df['close_prev'] > df['open_prev']) &                    # 第一天是阳线
        (df['close'] < df['open']) &                             # 第二天是阴线
        (df['body'] < df['body_prev'] * 0.5) &                   # 第二天实体小于第一天的一半
        (df['open'] < df['close_prev']) &                        # 第二天开盘价低于第一天收盘价
        (df['close'] > df['open_prev'])                          # 第二天收盘价高于第一天开盘价
    )
    df.loc[bearish_harami_condition, 'bearish_harami'] = 1
    
    # 在孕线形态之后添加反击线形态检测
    
    # 看涨反击线：第一天大阴线，第二天开盘继续下跌但收盘价接近前一天开盘价
    df['bullish_counter'] = 0
    bullish_counter_condition = (
        (df['close_prev'] < df['open_prev']) &                    # 第一天是阴线
        (df['body_prev'] > df['body'].mean()) &                   # 第一天是大阴线
        (df['open'] < df['close_prev']) &                         # 第二天以跳空低开
        (df['close'] > df['open']) &                             # 第二天是阳线
        (abs(df['close'] - df['open_prev']) < df['body_prev'] * 0.1)  # 第二天收盘价接近第一天开盘价
    )
    df.loc[bullish_counter_condition, 'bullish_counter'] = 1
    
    # 看跌反击线：第一天大阳线，第二天开盘继续上涨但收盘价接近前一天开盘价
    df['bearish_counter'] = 0
    bearish_counter_condition = (
        (df['close_prev'] > df['open_prev']) &                    # 第一天是阳线
        (df['body_prev'] > df['body'].mean()) &                   # 第一天是大阳线
        (df['open'] > df['close_prev']) &                         # 第二天以跳空高开
        (df['close'] < df['open']) &                             # 第二天是阴线
        (abs(df['close'] - df['open_prev']) < df['body_prev'] * 0.1)  # 第二天收盘价接近第一天开盘价
    )
    df.loc[bearish_counter_condition, 'bearish_counter'] = 1
    
    # 在反击线形态之后添加三法形态检测
    
    # 计算额外需要的特征
    df['body_prev3'] = df['body'].shift(3)
    df['body_prev4'] = df['body'].shift(4)
    df['close_prev3'] = df['close'].shift(3)
    df['close_prev4'] = df['close'].shift(4)
    df['open_prev3'] = df['open'].shift(3)
    df['open_prev4'] = df['open'].shift(4)
    
    # 上升三法：一根大阳线后跟三根小阴线，最后一根突破性大阳线
    df['rising_three'] = 0
    rising_three_condition = (
        # 第一天大阳线
        (df['close_prev4'] > df['open_prev4']) &                     # 第一天是阳线
        (df['body_prev4'] > df['body'].mean()) &                     # 第一天是大阳线
        
        # 中间三天小阴线，保持在第一天范围内
        (df['close_prev3'] < df['open_prev3']) &                     # 第二天是阴线
        (df['close_prev2'] < df['open_prev2']) &                     # 第三天是阴线
        (df['close_prev'] < df['open_prev']) &                       # 第四天是阴线
        
        # 中间三天的收盘价保持在第一天实体范围内
        (df['close_prev3'] > df['open_prev4']) &
        (df['close_prev2'] > df['open_prev4']) &
        (df['close_prev'] > df['open_prev4']) &
        
        # 最后一天突破性大阳线
        (df['close'] > df['open']) &                                 # 最后一天是阳线
        (df['close'] > df['close_prev4'])                           # 突破前高
    )
    df.loc[rising_three_condition, 'rising_three'] = 1
    
    # 下降三法：一根大阴线后跟三根小阳线，最后一根突破性大阴线
    df['falling_three'] = 0
    falling_three_condition = (
        # 第一天大阴线
        (df['close_prev4'] < df['open_prev4']) &                     # 第一天是阴线
        (df['body_prev4'] > df['body'].mean()) &                     # 第一天是大阴线
        
        # 中间三天小阳线，保持在第一天范围内
        (df['close_prev3'] > df['open_prev3']) &                     # 第二天是阳线
        (df['close_prev2'] > df['open_prev2']) &                     # 第三天是阳线
        (df['close_prev'] > df['open_prev']) &                       # 第四天是阳线
        
        # 中间三天的收盘价保持在第一天实体范围内
        (df['close_prev3'] < df['open_prev4']) &
        (df['close_prev2'] < df['open_prev4']) &
        (df['close_prev'] < df['open_prev4']) &
        
        # 最后一天突破性大阴线
        (df['close'] < df['open']) &                                 # 最后一天是阴线
        (df['close'] < df['close_prev4'])                           # 突破前低
    )
    df.loc[falling_three_condition, 'falling_three'] = 1
    
    # 在三法形态之后添加分手线形态检测
    
    # 看涨分手线：第一天阴线，第二天阳线，两天开盘价相同或接近
    df['bullish_separating'] = 0
    bullish_separating_condition = (
        (df['close_prev'] < df['open_prev']) &                    # 第一天是阴线
        (df['close'] > df['open']) &                             # 第二天是阳线
        (abs(df['open'] - df['open_prev']) < df['body'].mean() * 0.1) &  # 两天开盘价接近
        (df['close'] > df['close_prev']) &                       # 第二天收盘价高于第一天收盘价
        (df['body'] > df['body'].mean() * 0.8)                   # 确保实体足够大
    )
    df.loc[bullish_separating_condition, 'bullish_separating'] = 1
    
    # 看跌分手线：第一天阳线，第二天阴线，两天开盘价相同或接近
    df['bearish_separating'] = 0
    bearish_separating_condition = (
        (df['close_prev'] > df['open_prev']) &                    # 第一天是阳线
        (df['close'] < df['open']) &                             # 第二天是阴线
        (abs(df['open'] - df['open_prev']) < df['body'].mean() * 0.1) &  # 两天开盘价接近
        (df['close'] < df['close_prev']) &                       # 第二天收盘价低于第一天收盘价
        (df['body'] > df['body'].mean() * 0.8)                   # 确保实体足够大
    )
    df.loc[bearish_separating_condition, 'bearish_separating'] = 1
    
    # 更新要删除的中间计算列
    columns_to_drop = [
        'body', 'body_prev', 'body_prev2', 
        'close_prev', 'open_prev', 'close_prev2', 'open_prev2',
        'mid_point', 'mid_point_prev', 'mid_point_prev2',
        'body_prev3', 'body_prev4',
        'close_prev3', 'close_prev4',
        'open_prev3', 'open_prev4'
    ]
    df = df.drop(columns_to_drop, axis=1)
    
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