import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "请先安装 mamba-ssm 库: pip install causal-conv1d mamba-ssm"
    )

# =====================================================
# Mamba Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
#
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


class MambaBlock(nn.Module):
    """带残差连接和 Pre-LayerNorm 的标准 Mamba 模块"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        pred_len=30,
        d_model=64,
        num_layers=2,
        terrain_dim=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.terrain_dim = terrain_dim

        # 1. 输入时序投影 (42 -> d_model)
        self.embedding = nn.Linear(input_dim, d_model)

        # 2. Mamba 骨干网络堆叠
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # 3. 规范化与正则化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. 地形回归头 (直接回归到 [pred_len * 6]，严格保证通道排序)
        self.terrain_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * terrain_dim),
        )

        # 5. 权重初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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

        B = x.size(0)

        # 输入嵌入 [B, 200, d_model]
        h = self.embedding(x)

        # 通过 Mamba 状态空间层
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        # 提取汇聚了完整 200 帧因果历史信息的末帧隐藏状态 [B, d_model]
        feature = h[:, -1, :]
        feature = self.dropout(feature)

        # 回归到 [B, pred_len, terrain_dim]
        # 通道顺序严格保持: [ipsi_x, ipsi_y, ipsi_z, contra_x, contra_y, contra_z]
        terrain = self.terrain_head(feature)
        terrain = terrain.view(B, self.pred_len, self.terrain_dim)

        return {"terrain": terrain}


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 与 build_model("mamba", cfg) 传参完全兼容
    model = MambaTerrainRegression(
        input_dim=42,
        pred_len=30,
        d_model=64,
    ).to(device)

    dummy_imu = torch.randn(8, 200, 42).to(device)
    out = model(dummy_imu)

    print("Mamba 模型测试通过:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "输出维度与训练任务不一致！"