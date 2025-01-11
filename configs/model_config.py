class ModelConfig:
    # 模型参数
    HIDDEN_DIM = 512    # 隐藏层维度，决定了模型的容量和表达能力
    NUM_LAYERS = 8      # Transformer编码器/解码器的层数，层数越多模型越深
    NUM_HEADS = 8       # 多头注意力机制中的头数，每个头学习不同的特征关系
    DROPOUT = 0.3       # Dropout比率，用于防止过拟合，0.3表示30%的神经元会被随机关闭
    LOOKBACK = 50       # 回看窗口大小，表示模型考虑前50个时间步的数据
    
    # 训练参数
    BATCH_SIZE = 256  # 调回原来的大小
    NUM_EPOCHS = 10    # 训练轮数
    LEARNING_RATE = 1e-4  # 学习率
    ACCUMULATION_STEPS = 4  # 梯度累积步数
    
    # 数据参数
    DATA_PATH = './data/data.json'  # 数据文件路径
    MODEL_SAVE_PATH = 'best_model.pth'  # 模型保存路径 
    
    # 添加混合精度配置
    USE_AMP = True  # 启用混合精度
    GRAD_CLIP = 1.0  # 梯度裁剪阈值 
    
    # 优化内存使用的配置
    PREFETCH_FACTOR = 2  # 减少预加载数量
    PIN_MEMORY = True  # 使用固定内存 