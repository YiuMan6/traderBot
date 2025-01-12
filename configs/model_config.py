class ModelConfig:
    # 模型参数 - 考虑到数据规模，调整模型大小
    HIDDEN_DIM = 256    # 从512降到256，减少计算量
    NUM_LAYERS = 6      # 从8降到6，减少计算量
    NUM_HEADS = 8       # 保持不变
    DROPOUT = 0.2       # 从0.3降到0.2，加快训练
    LOOKBACK = 30       # 从50降到30，减少序列长度
    
    # 训练参数
    BATCH_SIZE = 512    # 从256增加到512，提高吞吐量
    NUM_EPOCHS = 10     
    LEARNING_RATE = 2e-4  # 略微提高学习率
    EARLY_STOPPING = 5  # 早停检查
    # 性能优化
    PREFETCH_FACTOR = 4    # 增加预加载数量
    PIN_MEMORY = True
    NUM_WORKERS = 4        # 增加工作进程数
    
    # 数据参数
    DATA_PATH = './data/data.json'  # 数据文件路径
    MODEL_SAVE_PATH = 'best_model.pth'  # 模型保存路径 
    
    # 添加混合精度配置
    USE_AMP = True  # 启用混合精度
    GRAD_CLIP = 1.0  # 梯度裁剪阈值 
    
    # 优化内存使用的配置
    PREFETCH_FACTOR = 2  # 减少预加载数量
    PIN_MEMORY = True  # 使用固定内存 