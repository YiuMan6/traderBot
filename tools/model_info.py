def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n模型可训练参数总量: {total_params:,}')
    return total_params