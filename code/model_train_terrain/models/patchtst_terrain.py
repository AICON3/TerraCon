import math
import torch
import torch.nn as nn

# =====================================================
# PatchTST Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
#
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, patch_num, d_model]
        return x + self.pe[:, : x.size(1)]


class PatchTSTTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        input_len=200,
        pred_len=30,
        patch_len=16,
        stride=8,
        terrain_dim=6,
        d_model=128,
        n_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.terrain_dim = terrain_dim

        # 1. 计算 Patch 数量: (200 - 16) // 8 + 1 = 24
        self.patch_num = (input_len - patch_len) // stride + 1

        # 2. Patch 局部特征映射 [input_dim * patch_len] -> [d_model]
        self.patch_embedding = nn.Linear(input_dim * patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # 3. Transformer 编码器主干
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.feature_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. 地形回归输出头 (汇聚各 Patch 特征映射到目标维度)
        self.terrain_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.patch_num * d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len * terrain_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """输入:

            x: [B, 200, 42] (IMU 序列)
        输出:
            dict: {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        B, T, D = x.shape

        # A. 时序切分 Patch: [B, T, D] -> [B, patch_num, patch_len * D]
        patches = (
            x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(B, self.patch_num, self.patch_len * D)
        )

        # B. 线性嵌入与位置编码
        h = self.patch_embedding(patches)  # [B, patch_num, d_model]
        h = self.pos_encoder(h)

        # C. Transformer 编码交互
        feature = self.encoder(h)
        feature = self.feature_norm(feature)
        feature = self.dropout(feature)

        # D. 回归未来地形参数 [B, pred_len, terrain_dim]
        terrain = self.terrain_head(feature)
        terrain = terrain.view(B, self.pred_len, self.terrain_dim)

        return {"terrain": terrain}


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    # 与 build_model("patchtst", cfg) 传参完全一致
    model = PatchTSTTerrainRegression(
        input_dim=42,
        input_len=200,
        pred_len=30,
        patch_len=16,
        stride=8,
    )

    x = torch.randn(8, 200, 42)
    out = model(x)

    print("PatchTST 验证成功:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "输出维度不符合预期！"