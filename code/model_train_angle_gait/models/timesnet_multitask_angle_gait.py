import torch
import torch.nn as nn
import torch.nn.functional as F


def FFT_for_Period(x, k=5):
    """Extract dominant sequence periods and their spectral-amplitude weights

    x: [B, T, C]

    Returns:
        period:        list[int], length k, containing periods of dominant components
        period_weight: [B, k], spectral amplitudes corresponding to each period,
                       Used later for adaptive softmax-weighted fusion
    """
    xf = torch.fft.rfft(x, dim=1)
    amplitude = torch.abs(xf).mean(dim=-1)      # [B, T//2 + 1]

    frequency = amplitude.mean(dim=0)           # [T//2 + 1]
    # Clone to avoid in-place modification of the autograd graph
    frequency = frequency.clone()
    frequency[0] = 0.0                          # Suppress the DC component

    # Ensure k does not exceed the number of frequency bins
    k = max(min(k, frequency.size(0) - 1), 1)
    _, top_list = torch.topk(frequency, k)

    period = []
    for top in top_list.detach().cpu().tolist():
        # Guard against division by zero
        top = max(int(top), 1)
        period.append(max(x.shape[1] // top, 1))

    period_weight = amplitude[:, top_list]      # [B, k]

    return period, period_weight


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
        periods, period_weight = FFT_for_Period(x, self.top_k)
        res = []

        for period in periods:
            length = T
            if length % period != 0:
                length = (length // period + 1) * period
                pad_len = length - T
                # Use new_zeros to preserve input dtype and device
                # (torch.zeros defaults to float32, which can break cat under mixed precision)
                padding = x.new_zeros(B, pad_len, C)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x

            # Transform to 2D periodic space: [B, length, C] -> [B, C, length//period, period]
            out = out.reshape(B, length // period, period, C).permute(
                0, 3, 1, 2
            )
            out = self.conv(out)
            # Restore 1D sequence: [B, C, length//period, period] -> [B, length, C]
            out = out.permute(0, 2, 3, 1).reshape(B, length, C)
            out = out[:, :T, :]
            res.append(out)

        # Adaptive weighted fusion over periods
        # Weight periods with softmax over their spectral amplitudes instead of a simple mean
        res = torch.stack(res, dim=-1)                      # [B, T, C, k]
        period_weight = F.softmax(
            period_weight.float(), dim=1
        )                                                   # [B, k]
        period_weight = period_weight.to(res.dtype).unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * period_weight, dim=-1)        # [B, T, C]
        return res + x


class TimesNetMultiTask(nn.Module):
    """TimesNet multitask model (angle regression and gait classification)

    Architecture:
        Linear Embedding -> TimesBlock x num_layers -> LayerNorm
        -> temporal length projection -> {angle_head, gait_head}

    Input:
        x: [B, input_len, input_dim]   (default [B, 200, 42])
    Output:
        {"angle": [B, pred_len, angle_dim],
         "gait":  [B, pred_len, gait_dim]}
    """

    def __init__(
        self,
        input_len=200,
        pred_len=30,
        input_dim=42,
        angle_dim=8,
        gait_dim=2,
        d_model=64,
        d_ff=128,
        num_layers=2,
        top_k=5,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.angle_dim = angle_dim
        self.gait_dim = gait_dim

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

        # 3. Map the entire input_len history to the target pred_len
        self.predict_linear = nn.Linear(input_len, pred_len)

        # 4. Angle regression head (angle_dim outputs)
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, angle_dim),
        )

        # 5. Gait classification head (gait_dim logits)
        self.gait_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, gait_dim),
        )

    def forward(self, x, **kwargs):
        """Input:
            x: [B, input_len, input_dim]
        Output:
            {"angle": [B, pred_len, angle_dim],
             "gait":  [B, pred_len, gait_dim]}
        """
        # The dataset has already applied nan_to_num and scaling by 1/100,
        # so do not call torch.nan_to_num on GPU tensors to avoid
        # illegal memory access with some PyTorch/CUDA versions.
        x = x.contiguous()
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        B, T, _ = x.shape

        if T != self.input_len:
            raise ValueError(
                f"Input length {T} differs from configured input_len {self.input_len}"
            )

        # A. Linear embedding
        h = self.embedding(x)
        h = self.dropout(h)

        # B. Periodic 2D convolution transformation
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)  # [B, input_len, d_model]

        # C. Project across time to the prediction horizon
        # [B, input_len, d_model] -> [B, d_model, input_len]
        # -> [B, d_model, pred_len] -> [B, pred_len, d_model]
        h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)

        # D. Multitask outputs
        angle = self.angle_head(h)  # [B, pred_len, angle_dim]
        gait = self.gait_head(h)    # [B, pred_len, gait_dim]

        return {"angle": angle, "gait": gait}


if __name__ == "__main__":
    model = TimesNetMultiTask(
        input_len=200,
        pred_len=30,
        input_dim=42,
        d_model=64,
        d_ff=128,
        num_layers=2,
    )
    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)
    print("angle:", out["angle"].shape)
    print("gait:", out["gait"].shape)
    assert out["angle"].shape == (8, 30, 8)
    assert out["gait"].shape == (8, 30, 2)
    print("TimesNet validation passed")
