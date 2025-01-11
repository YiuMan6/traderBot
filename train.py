import torch
import torch.nn as nn
from torch.optim import AdamW
from model import TransformerModel
from dataLoader import load_data
from tqdm import tqdm
from tools.device import get_device
from configs.model_config import ModelConfig
from tools.model_info import count_parameters
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

## source ~/.bashrc
##conda activate pytorchEnv

def analyze_model_parameters(model):
    """分析模型参数的数据类型和统计信息"""
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
    print(f"\n估计内存使用: {param_stats['memory_usage']:.2f} MB")
    
    return param_types, param_stats

def clear_gpu_memory():
    """清理GPU显存"""
    torch.cuda.empty_cache()
    
def print_gpu_memory():
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

def train_model(model, train_loader, num_epochs=ModelConfig.NUM_EPOCHS, 
                learning_rate=ModelConfig.LEARNING_RATE):
    
    # 在关键点监控显存
    print("Initial GPU memory:")
    print_gpu_memory()
    
    # 只在开始时进行一次分析
    print("Initial model analysis:")
    analyze_model_parameters(model)
    
    # 移除这部分重复分析
    # print("Before mixed precision:")
    # analyze_model_parameters(model)
    # with autocast():
    #     print("\nWith mixed precision:")
    #     analyze_model_parameters(model)
    
    device = get_device()
    model = model.to(device)

    count_parameters(model)
    
    # 使用AdamW优化器
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    # 添加 tensorboard writer
    writer = SummaryWriter('runs/experiment_1')
    
    # 存储训练历史
    history = {
        'loss': [],
        'accuracy': [],
        'attention_patterns': []
    }
    
    # 1. 初始化混合精度训练工具
    scaler = GradScaler()
    
    # 添加额外的监控
    nan_count = 0
    inf_count = 0
    
    for epoch in range(num_epochs):
        # 每个epoch开始时清理显存
        clear_gpu_memory()
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (batch_x, batch_y) in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 2. 使用 autocast 进行前向传播和损失计算
            with autocast():
                outputs = model(batch_x)
                
                # 监控数值问题
                if torch.isnan(outputs).any():
                    nan_count += 1
                if torch.isinf(outputs).any():
                    inf_count += 1
                
                if batch_idx % 100 == 0:
                    print(f"NaN count: {nan_count}, Inf count: {inf_count}")
                
                loss = criterion(outputs, batch_y)
            
            # 3. 使用 scaler 进行反向传播
            scaler.scale(loss).backward()
            
            # 4. 使用 scaler 更新参数
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            total_loss += loss.item()
            acc = 100. * correct / total
            
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs}')
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{acc:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        epoch_acc = 100. * correct / total
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), ModelConfig.MODEL_SAVE_PATH)
            print(f'\n保存最佳模型，准确率: {best_acc:.2f}%')
        
        print(f'\nEpoch {epoch+1}: '
              f'Loss = {total_loss/len(train_loader):.4f}, '
              f'Accuracy = {epoch_acc:.2f}%, '
              f'Best Accuracy = {best_acc:.2f}%')
        
        # 记录每个 epoch 的指标
        writer.add_scalar('Loss/train', total_loss/len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        history['loss'].append(total_loss/len(train_loader))
        history['accuracy'].append(epoch_acc)

        # 修改分析部分
        if epoch % 5 == 0:
            # 分析前清理显存
            clear_gpu_memory()
            analyze_predictions(model, train_loader, epoch)
            
            if epoch % 10 == 0:
                clear_gpu_memory()
                importance = analyze_feature_importance(model, train_loader)
                plt.figure(figsize=(10, 5))
                plt.bar(range(len(importance)), importance.numpy())
                plt.title(f'Feature Importance - Epoch {epoch}')
                plt.xlabel('Feature Index')
                plt.ylabel('Importance Score')
                plt.savefig(f'feature_importance_epoch_{epoch}.png')
                plt.close()

        if epoch % 5 == 0:
            print(f"\nEpoch {epoch} GPU memory:")
            print_gpu_memory()

def analyze_predictions(model, data_loader, epoch):
    """分析模型的预测模式"""
    device = next(model.parameters()).device  # 获取模型所在的设备
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # 将数据移动到正确的设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            _, preds = torch.max(outputs.data, 1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # 分析预测分布
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
    plt.close()

def analyze_feature_importance(model, data_loader):
    """分析输入特征的重要性"""
    device = next(model.parameters()).device  # 获取模型所在的设备
    feature_importance = torch.zeros(model.feature_dim, device=device)
    
    for batch_x, _ in data_loader:
        # 将数据移动到正确的设备
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        outputs = model(batch_x)
        loss = outputs.mean()
        loss.backward()
        
        # 计算每个特征的重要性
        # 对所有维度取平均，只保留特征维度
        importance = torch.abs(batch_x.grad).mean(dim=(0, 1))  # 平均batch和时间维度
        feature_importance += importance
    
    # 移动到CPU并归一化
    feature_importance = feature_importance.cpu() / len(data_loader)
    
    return feature_importance

def monitor_training(model, outputs, loss):
    """监控训练过程中的关键指标"""
    stats = {
        'memory_allocated': torch.cuda.memory_allocated() / 1e9,  # GB
        'max_memory': torch.cuda.max_memory_allocated() / 1e9,
        'nan_count': torch.isnan(outputs).sum().item(),
        'inf_count': torch.isinf(outputs).sum().item(),
        'loss_value': loss.item(),
        'grad_norm': get_grad_norm(model)
    }
    return stats

def get_grad_norm(model):
    """计算梯度范数"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

if __name__ == "__main__":
    # 1. 直接使用配置类中的参数
    
    # 2. 加载数据
    data = load_data(ModelConfig.DATA_PATH, 
                    batch_size=ModelConfig.BATCH_SIZE, 
                    lookback=ModelConfig.LOOKBACK)
    
    # 3. 创建模型
    model = TransformerModel(
        feature_dim=data['feature_dim'],
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        num_heads=ModelConfig.NUM_HEADS,
        lookback=ModelConfig.LOOKBACK,
        dropout=ModelConfig.DROPOUT
    )
    
    # 4. 训练模型
    train_model(
        model=model,
        train_loader=data['train_loader'],
        num_epochs=ModelConfig.NUM_EPOCHS,
        learning_rate=ModelConfig.LEARNING_RATE
    )