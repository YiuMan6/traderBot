import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, num_heads, lookback=30, dropout=0.1):
        super().__init__()
        
        # 使用内存效率更高的设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 保存特征维度
        self.feature_dim = feature_dim
        
        # 1. 更强的特征提取
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(lookback),  # 添加批量归一化
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. 添加特征嵌入
        self.feature_embedding = nn.Parameter(torch.randn(1, feature_dim, hidden_dim))
        self.position_embedding = nn.Parameter(torch.randn(lookback, 1, hidden_dim))
        
        # 3. 更深的Transformer
        encoder_layers = []
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            encoder_layers.append(layer)
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # 4. 更复杂的输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 连接最后时间步和平均池化
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 存储最后一次的注意力权重
        self.attention_weights = None

    def forward(self, x):
        # 1. 特征提取
        x = self.input_projection(x)
        
        # 2. 添加嵌入
        x = x + self.position_embedding.transpose(0, 1)
        
        # 3. Transformer处理
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 4. 多特征融合
        last_hidden = x[:, -1, :]  # 最后时间步
        avg_hidden = torch.mean(x, dim=1)  # 平均池化
        combined = torch.cat([last_hidden, avg_hidden], dim=1)  # 特征连接
        
        # 5. 输出预测
        return self.output_layer(combined)
    
    def get_attention_weights(self, x):
        """获取注意力权重的方法"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            _ = self(x)  # 运行前向传播以获取注意力权重
            return self.attention_weights

# 自定义 TransformerEncoder 来获取注意力权重
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, return_attention=False):
        output = src
        attention_weights = []
        
        for layer in self.layers:
            output, attn = layer(output)
            if return_attention:
                attention_weights.append(attn)
        
        if return_attention:
            # 返回最后一层的注意力权重
            return output, attention_weights[-1]
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # ... 其他层的定义 ...
        
    def forward(self, src):
        # 获取注意力输出和权重
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        # ... 处理其他层 ...
        return output, attn_weights