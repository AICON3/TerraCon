import torch
import torch.nn as nn

# =====================================================
# LSTM Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
#
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


class LSTMTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        hidden_dim=128,
        num_layers=2,
        pred_len=30,
        terrain_dim=6,
        dropout=0.2,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.terrain_dim = terrain_dim

        # 1. 基础时序编码器: 双向或多层 LSTM (若 num_layers > 1 支持 dropout)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. 状态规范化与正则化
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. 时序长度投影: 将 LSTM 最后一帧隐状态直接展开并投影到 [pred_len, terrain_dim]
        # 避免仅依赖尾部 slice 导致的因果特征泄露/截断偏差
        self.terrain_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * terrain_dim),
        )

        # 4. 权重参数初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def forward(self, x):
        """输入:

            x: [B, 200, 42] (IMU 序列)
        输出:
            dict: {"terrain": [B, 30, 6]}
        """
        # 异常值清理
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        B = x.size(0)

        # LSTM 时序特征提取: out: [B, T, hidden_dim]
        out, _ = self.lstm(x)

        # 提取聚合了整个 200 帧完整历史动力学的末尾隐状态 [B, hidden_dim]
        last_hidden = out[:, -1, :]
        last_hidden = self.norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # 映射到 [B, pred_len * terrain_dim] 并重塑为 [B, pred_len, terrain_dim]
        terrain = self.terrain_head(last_hidden)
        terrain = terrain.view(B, self.pred_len, self.terrain_dim)

        return {"terrain": terrain}


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    # 与 build_model("lstm", cfg) 传参一致
    model = LSTMTerrainRegression(
        input_dim=42,
        hidden_dim=128,
        num_layers=2,
        pred_len=30,
        terrain_dim=6,
        dropout=0.2,
    )

    x = torch.randn(8, 200, 42)
    out = model(x)

    print("LSTM 验证成功:")
    print("terrain output shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])