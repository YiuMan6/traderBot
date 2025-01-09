import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, num_heads, lookback=30, dropout=0.1):
        super().__init__()
        
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