def analyze_model_parameters(model):
    param_types = {}
    param_stats = {
        'total_params': 0,
        'trainable_params': 0,
        'param_sizes': {},
        'memory_usage': 0
    }
    
    for name, param in model.named_parameters():
        # 获取参数类型
        dtype = str(param.dtype)
        param_types[dtype] = param_types.get(dtype, 0) + param.numel()
        
        # 统计参数信息
        param_stats['total_params'] += param.numel()
        if param.requires_grad:
            param_stats['trainable_params'] += param.numel()
        
        # 记录每层参数大小
        param_stats['param_sizes'][name] = {
            'shape': list(param.shape),
            'num_params': param.numel()
        }
        
        # 估算内存使用
        memory_bits = param.numel() * param.element_size()
        param_stats['memory_usage'] += memory_bits
    
    # 转换为MB
    param_stats['memory_usage'] = param_stats['memory_usage'] / (1024 * 1024)
    
    print("\n=== 模型参数分析 ===")
    print(f"总参数量: {param_stats['total_params']:,}")
    print(f"可训练参数量: {param_stats['trainable_params']:,}")
    print("\n参数数据类型分布:")
    for dtype, count in param_types.items():
        percentage = count / param_stats['total_params'] * 100
        print(f"{dtype}: {count:,} ({percentage:.2f}%)")
    
    return param_types, param_stats