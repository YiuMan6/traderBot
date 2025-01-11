import torch
from pattern_monitor import monitor_patterns
from model import TransformerModel
from dataLoader import load_data
from configs.model_config import ModelConfig

def run_pattern_analysis():
    # 1. 加载数据
    data = load_data(ModelConfig.DATA_PATH, 
                    batch_size=ModelConfig.BATCH_SIZE, 
                    lookback=ModelConfig.LOOKBACK)
    
    # 2. 加载训练好的模型
    model = TransformerModel(
        feature_dim=data['feature_dim'],
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        num_heads=ModelConfig.NUM_HEADS,
        lookback=ModelConfig.LOOKBACK,
        dropout=ModelConfig.DROPOUT
    )
    
    # 3. 加载保存的模型权重（使用更安全的加载方式）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(
        torch.load(
            ModelConfig.MODEL_SAVE_PATH,
            map_location=device,
            weights_only=True  # 只加载权重，更安全
        )
    )
    model.eval()
    
    # 4. 获取所有K线形态的列名
    pattern_columns = [col for col in data['feature_names'] if 'pattern' in col.lower()]
    print("\n可用的K线形态:", pattern_columns)
    
    # 5. 分析每种形态
    for pattern in pattern_columns:
        results = monitor_patterns(model, data, pattern)
        
        # 保存结果到CSV，并添加时间戳避免覆盖
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results.to_csv(f'{pattern}_analysis_{timestamp}.csv', index=False)
        print(f"\n结果已保存到: {pattern}_analysis_{timestamp}.csv")

if __name__ == "__main__":
    try:
        run_pattern_analysis()
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}") 