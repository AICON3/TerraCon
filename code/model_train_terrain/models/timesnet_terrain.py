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
    """Extract spectral components for dominant sequence periods

    x: [B, T, C]
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency = torch.abs(xf).mean(dim=0).mean(dim=-1)
    frequency[0] = 0.0  # Suppress the DC component

    # Ensure k does not exceed the number of frequency bins
    k = min(k, frequency.size(0) - 1)
    _, top_list = torch.topk(frequency, k)
    top_list = top_list.detach().cpu().numpy()

    period = []
    for top in top_list:
        # Guard against division by zero
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

            # Transform to 2D periodic space: [B, 2D_T, 2D_P, C] -> [B, C, 2D_T, 2D_P]
            out = out.reshape(B, length // period, period, C).permute(
                0, 3, 1, 2
            )
            out = self.conv(out)
            # Restore 1D sequence: [B, C, 2D_T, 2D_P] -> [B, length, C]
            out = out.permute(0, 2, 3, 1).reshape(B, length, C)
            out = out[:, :T, :]
            res.append(out)

        # Weighted fusion over periods
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

        # 1. Local feature embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # 2. Stacked TimesBlocks
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

        # 3. Map all 200 input frames to the target pred_len (30)
        self.predict_linear = nn.Linear(input_len, pred_len)

        # 4. Terrain regression head (six outputs)
        self.terrain_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, terrain_dim),
        )

    def forward(self, x):
        """Input:

            x: [B, 200, 42]
        Output:
            dict: {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        # A. Linear embedding
        h = self.embedding(x)
        h = self.dropout(h)

        # B. Periodic 2D convolution transformation
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)  # [B, 200, d_model]

        # C. Project across time while preserving the full historical context
        # [B, 200, d_model] -> [B, d_model, 200] -> [B, d_model, 30] -> [B, 30, d_model]
        h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)

        # D. Regress terrain parameters [B, 30, 6]
        terrain = self.terrain_head(h)

        return {"terrain": terrain}


if __name__ == "__main__":
    # Arguments are compatible with build_model("timesnet", cfg)
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

    print("TimesNet validation passed:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "Output shape does not meet training requirements!"