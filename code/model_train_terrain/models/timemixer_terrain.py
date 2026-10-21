import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# TimeMixer Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
#
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


class PastDecomposableMixing(nn.Module):
    """时序多尺度解构混合块 (TimeMixer 核心单元)"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 时序维度混合 (Token Mixing)
        self.time_mixing = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.time_mixing(self.norm(x))


class TimeMixerTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        pred_len=30,
        d_model=64,
        terrain_dim=6,
        num_scales=3,
        down_sampling_window=2,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.terrain_dim = terrain_dim
        self.num_scales = num_scales
        self.down_sampling_window = down_sampling_window

        # 1. 各尺度的输入线性投影 (42 -> d_model)
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, d_model) for _ in range(num_scales)
        ])

        # 2. 各尺度的 Past Decomposable Mixing 块
        self.mixers = nn.ModuleList([
            nn.Sequential(
                PastDecomposableMixing(d_model, dropout),
                PastDecomposableMixing(d_model, dropout),
            )
            for _ in range(num_scales)
        ])

        # 3. 多尺度特征融合层 (拼接多个尺度的池化特征)
        total_dim = d_model * num_scales
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.GELU(),
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
        )

        # 4. 地形回归输出头 (严格对应 [pred_len, terrain_dim])
        self.terrain_head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * terrain_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_scale(self, x, scale_idx):
        # 投影并进行时序特征混合
        h = self.encoders[scale_idx](x)
        h = self.mixers[scale_idx](h)
        # 全局时序平均池化 [B, T_scale, d_model] -> [B, d_model]
        return h.mean(dim=1)

    def forward(self, x):
        """输入:

            x: [B, 200, 42] (IMU 序列)
        输出:
            dict: {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim != 3:
            raise ValueError("Input must be [B, T, F]")

        # 维度容错校验
        if x.size(-1) != self.input_dim:
            if x.size(1) == self.input_dim:
                x = x.transpose(1, 2)
            else:
                raise ValueError(
                    f"输入特征维度 {x.shape} 与设定 input_dim {self.input_dim} 不一致"
                )

        B = x.size(0)
        pooled = []
        current = x

        # 多尺度降采样与编码
        for level in range(self.num_scales):
            pooled.append(self._encode_scale(current, level))

            # 下采样生成粗粒度时序: [B, T, F] -> [B, F, T] -> avg_pool -> [B, T_new, F]
            if level < self.num_scales - 1:
                current = F.avg_pool1d(
                    current.transpose(1, 2),
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                ).transpose(1, 2)

        # 跨尺度特征融合 [B, d_model * num_scales]
        multi_scale_feature = torch.cat(pooled, dim=-1)
        feature = self.fusion(multi_scale_feature)

        # 地形回归并重塑为 [B, pred_len, terrain_dim]
        terrain = self.terrain_head(feature)
        terrain = terrain.view(B, self.pred_len, self.terrain_dim)

        return {"terrain": terrain}


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    # 与 build_model("timemixer", cfg) 传参完全一致
    model = TimeMixerTerrainRegression(
        input_dim=42,
        pred_len=30,
        d_model=64,
    )

    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)

    print("TimeMixer 验证通过:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "输出维度不一致！"