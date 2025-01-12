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
    
def train_model(model, train_loader, val_loader, num_epochs):
    best_val_loss = float('inf')
    patience = ModelConfig.EARLY_STOPPING
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader)
        
        # 监控指标
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def train_epoch(model, train_loader):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=ModelConfig.LEARNING_RATE)
    scaler = GradScaler()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
        total_loss += loss.item()
        
        # 更新进度条
        acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{total_loss/len(train_loader):.4f}', 
            'acc': f'{acc:.2f}%'
        })
    
    return total_loss/len(train_loader), acc

def validate(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            total_loss += loss.item()
    
    return total_loss/len(val_loader), 100. * correct / total

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
        val_loader=data['val_loader'],
        num_epochs=ModelConfig.NUM_EPOCHS
    )