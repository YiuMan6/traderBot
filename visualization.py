import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime

def visualize_patterns(df, pattern_name=None, window_size=None):
    """在K线图上可视化标注K线形态"""
    # 准备数据
    df = df.copy()
    df.set_index('time', inplace=True)
    
    # 获取所有形态列
    pattern_cols = [col for col in df.columns if 'pattern' in col]
    print("\n可用的形态:")
    for col in pattern_cols:
        count = df[col].sum()
        print(f"{col}: 出现 {count} 次")
    
    if pattern_name:
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # 使用所有数据而不是只用最后window_size个
        df_plot = df.copy()
        df_plot['index'] = range(len(df_plot))
        
        # 计算K线颜色
        colors = np.where(df_plot['close'] >= df_plot['open'], 'red', 'green')
        
        # 绘制K线
        ax1.vlines(df_plot['index'], df_plot['low'], df_plot['high'], 
                  color=colors, linewidth=0.5)  # 减小线宽以适应更多数据
        ax1.vlines(df_plot['index'], df_plot['open'], df_plot['close'], 
                  color=colors, linewidth=2)
        
        # 标记形态出现的位置
        pattern_points = df_plot[df_plot[pattern_name] == 1]
        if not pattern_points.empty:
            ax1.scatter(pattern_points['index'], 
                       pattern_points['high'], 
                       marker='^', 
                       color='blue', 
                       s=50,  # 减小标记大小
                       label=pattern_name)
            
            # 只在重要位置添加文本标签，避免过度拥挤
            for idx, row in pattern_points.iterrows():
                if np.random.random() < 0.1:  # 随机显示10%的标签
                    ax1.text(row['index'], 
                            row['high'], 
                            pattern_name.replace('_pattern', ''),
                            rotation=45,
                            fontsize=6)
        
        # 绘制成交量
        ax2.bar(df_plot['index'], df_plot['volume'], 
                color=colors, alpha=0.5, width=1)
        
        # 设置标题和标签
        ax1.set_title(f'{pattern_name} 形态识别 (总数据: {len(df_plot)}条, 形态出现: {len(pattern_points)}次)')
        ax1.set_ylabel('价格')
        ax2.set_ylabel('成交量')
        
        # 设置x轴标签
        num_ticks = 20  # 显示20个时间标签
        xticks = np.linspace(0, len(df_plot)-1, num_ticks, dtype=int)
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)
        
        # 使用实际的时间作为标签
        date_labels = df_plot.index[xticks].strftime('%Y-%m-%d\n%H:%M')
        ax1.set_xticklabels(date_labels, rotation=45)
        ax2.set_xticklabels(date_labels, rotation=45)
        
        # 添加网格
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        ax1.legend()
        
        plt.tight_layout()
        plt.savefig(f'{pattern_name}_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n图表已保存为: {pattern_name}_visualization.png")
        
        # 打印形态出现的具体时间点
        print(f"\n{pattern_name} 出现的时间点:")
        for idx, row in pattern_points.iterrows():
            print(f"时间: {idx}, 价格: {row['close']:.2f}")

def analyze_features(df):
    """分析所有特征"""
    print("\n特征分析:")
    print("="*50)
    
    # 1. 基本特征
    print("\n基本特征:")
    basic_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in basic_cols:
        if col in df.columns:
            print(f"{col}: 范围 [{df[col].min():.2f}, {df[col].max():.2f}]")
    
    # 2. 形态特征
    print("\nK线形态:")
    pattern_cols = [col for col in df.columns if 'pattern' in col]
    for col in pattern_cols:
        occurrences = df[col].sum()
        if occurrences > 0:
            pattern_dates = df[df[col] == 1].index
            print(f"\n{col}:")
            print(f"总共出现: {occurrences} 次")
            print("出现时间点示例:")
            for date in pattern_dates[:5]:  # 显示前5个时间点
                print(f"- {date}")
    
    return {
        'basic_features': basic_cols,
        'pattern_features': pattern_cols
    }

if __name__ == "__main__":
    # 使用示例
    from dataLoader import load_data
    from configs.model_config import ModelConfig
    
    # 加载数据
    data = load_data(ModelConfig.DATA_PATH, 
                    batch_size=ModelConfig.BATCH_SIZE, 
                    lookback=ModelConfig.LOOKBACK)
    
    df = data['df']
    
    # 分析特征
    features = analyze_features(df)
    
    # 可视化每种形态
    for pattern in features['pattern_features']:
        visualize_patterns(df, pattern) 