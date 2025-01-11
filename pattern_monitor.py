import torch
import pandas as pd
from datetime import datetime

def monitor_patterns(model, data, pattern_name='hammer'):
    """监控特定K线形态的出现和表现"""
    df = data['df']
    device = next(model.parameters()).device
    
    # 找出所有该形态出现的时间点
    pattern_occurrences = []
    
    # 遍历数据
    for i in range(len(df) - data['lookback']):
        # 获取当前时间窗口的数据
        window = df.iloc[i:i + data['lookback']]
        
        # 检查最后一个时间点是否出现了目标形态
        if window[pattern_name].iloc[-1] == 1:
            # 获取模型预测
            with torch.no_grad():
                # 准备输入数据
                input_data = torch.FloatTensor(data['scaler'].transform(
                    window[data['feature_names']].values
                )).unsqueeze(0).to(device)
                
                # 获取预测
                output = model(input_data)
                prob = torch.softmax(output, dim=1)
                prediction = prob.argmax().item()
                confidence = prob.max().item()
            
            # 记录这次出现的详细信息
            actual_return = df['future_return'].iloc[i + data['lookback']]
            pattern_occurrences.append({
                'time': df['time'].iloc[i + data['lookback']],
                'price': df['close'].iloc[i + data['lookback']],
                'prediction': '上涨' if prediction == 1 else '下跌',
                'confidence': confidence,
                'actual_return': actual_return,
                'correct': (prediction == 1 and actual_return > 0) or 
                          (prediction == 0 and actual_return <= 0)
            })
    
    # 转换为DataFrame并打印结果
    results_df = pd.DataFrame(pattern_occurrences)
    print(f"\n{pattern_name} 形态分析结果:")
    print(f"总共出现次数: {len(results_df)}")
    print(f"预测准确率: {results_df['correct'].mean():.2f}")
    print("\n具体出现时间点:")
    for _, row in results_df.iterrows():
        print("\n" + "="*50)
        print(f"时间: {row['time']}")
        print(f"价格: {row['price']:.2f}")
        print(f"模型预测: {row['prediction']} (置信度: {row['confidence']:.2f})")
        print(f"实际收益率: {row['actual_return']:.2f}%")
        print(f"预测结果: {'✓ 正确' if row['correct'] else '✗ 错误'}")
    
    return results_df

# 使用示例
if __name__ == "__main__":
    # 加载训练好的模型
    model.eval()
    
    # 监控多个形态
    patterns_to_monitor = ['hammer', 'doji', 'engulfing', 'morning_star']
    for pattern in patterns_to_monitor:
        results = monitor_patterns(model, data, pattern) 