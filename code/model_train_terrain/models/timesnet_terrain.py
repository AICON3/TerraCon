import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# TimesNet Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


def FFT_for_Period(x, k=5):
    """提取时序主要周期的频率分量

    x: [B, T, C]
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency = torch.abs(xf).mean(dim=0).mean(dim=-1)
    frequency[0] = 0.0  # 压制直流分量

    # 保证 k 不超过频域采样点总数
    k = min(k, frequency.size(0) - 1)
    _, top_list = torch.topk(frequency, k)
    top_list = top_list.detach().cpu().numpy()

    period = []
    for top in top_list:
        # 防御除零风险
        top = max(int(top), 1)
        p = max(x.shape[1] // top, 1)
        period.append(p)

    return period


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_kernels=4):
        super().__init__()
        kernels = [1, 3, 5, 7]
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernels[i]),
                padding=(0, kernels[i] // 2),
            )
            for i in range(num_kernels)
        ])

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return torch.mean(torch.stack(res, dim=0), dim=0)


class TimesBlock(nn.Module):

    def __init__(self, seq_len, d_model, d_ff=128, top_k=5):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k

        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff),
            nn.GELU(),
            InceptionBlock(d_ff, d_model),
        )

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        periods = FFT_for_Period(x, self.top_k)
        res = []

        for period in periods:
            length = T
            if length % period != 0:
                length = (length // period + 1) * period
                pad_len = length - T
                padding = torch.zeros(B, pad_len, C, device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x

            # 变换到 2D 周期空间: [B, 2D_T, 2D_P, C] -> [B, C, 2D_T, 2D_P]
            out = out.reshape(B, length // period, period, C).permute(
                0, 3, 1, 2
            )
            out = self.conv(out)
            # 还原为 1D 时序: [B, C, 2D_T, 2D_P] -> [B, length, C]
            out = out.permute(0, 2, 3, 1).reshape(B, length, C)
            out = out[:, :T, :]
            res.append(out)

        # 多周期加权融合
        res = torch.stack(res, dim=-1).mean(dim=-1)
        return res + x


class TimesNetTerrainRegression(nn.Module):

    def __init__(
        self,
        input_len=200,
        pred_len=30,
        input_dim=42,
        terrain_dim=6,
        d_model=64,
        d_ff=128,
        num_layers=2,
        top_k=5,
        dropout=0.1,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.terrain_dim = terrain_dim

        # 1. 局部特征嵌入
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # 2. TimesBlock 堆叠
        self.blocks = nn.ModuleList([
            TimesBlock(
                seq_len=input_len,
                d_model=d_model,
                d_ff=d_ff,
                top_k=top_k,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 3. 时序长度投影: 将完整的 200 帧聚合映射到目标 pred_len (30)
        self.predict_linear = nn.Linear(input_len, pred_len)

        # 4. 地形回归头 (纯净 6 维输出)
        self.terrain_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, terrain_dim),
        )

    def forward(self, x):
        """输入:

            x: [B, 200, 42]
        输出:
            dict: {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        # A. 线性嵌入
        h = self.embedding(x)
        h = self.dropout(h)

        # B. 周期性 2D 卷积转换
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)  # [B, 200, d_model]

        # C. 跨时序投影到预测步长 (保持全局历史信息完整性)
        # [B, 200, d_model] -> [B, d_model, 200] -> [B, d_model, 30] -> [B, 30, d_model]
        h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)

        # D. 回归地形参数 [B, 30, 6]
        terrain = self.terrain_head(h)

        return {"terrain": terrain}


if __name__ == "__main__":
    # 与 build_model("timesnet", cfg) 参数完全兼容
    model = TimesNetTerrainRegression(
        input_len=200,
        pred_len=30,
        input_dim=42,
        d_model=64,
        d_ff=128,
        num_layers=2,
    )

    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)

    print("TimesNet 验证通过:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "输出维度与训练要求不符！"