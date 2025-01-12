import torch
import torch.nn as nn
from torch.optim import AdamW
from model import TransformerModel
from dataLoader import load_data
from tqdm import tqdm
from tools.device import get_device,clear_gpu_memory
from configs.model_config import ModelConfig
from tools.model_info import analyze_model_parameters
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

## source ~/.bashrc
##conda activate pytorchEnv
    
def train_model(model, train_loader, num_epochs=ModelConfig.NUM_EPOCHS, 
                learning_rate=ModelConfig.LEARNING_RATE):
    
    device = get_device()
    analyze_model_parameters(model)
    model = model.to(device)
    
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
            # analyze_predictions(model, train_loader, epoch)
            
            # if epoch % 10 == 0:
            #     clear_gpu_memory()
            #     importance = analyze_feature_importance(model, train_loader)
            #     plt.figure(figsize=(10, 5))
            #     plt.bar(range(len(importance)), importance.numpy())
            #     plt.title(f'Feature Importance - Epoch {epoch}')
            #     plt.xlabel('Feature Index')
            #     plt.ylabel('Importance Score')
            #     plt.savefig(f'feature_importance_epoch_{epoch}.png')
            #     plt.close()

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