import torch
import torch.nn as nn
from torch.optim import AdamW
from model import TransformerModel
from dataLoader import load_data
from tqdm import tqdm
from tools.device import get_device
from configs.model_config import ModelConfig
from tools.model_info import count_parameters

def train_model(model, train_loader, num_epochs=ModelConfig.NUM_EPOCHS, 
                learning_rate=ModelConfig.LEARNING_RATE, 
                accumulation_steps=ModelConfig.ACCUMULATION_STEPS):

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
        steps_per_epoch=len(train_loader)//accumulation_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (batch_x, batch_y) in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss = loss / accumulation_steps  # 缩放损失
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            total_loss += loss.item() * accumulation_steps
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