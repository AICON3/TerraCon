import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Causal convolution branch 1
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Causal convolution branch 2
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual alignment
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = x

        # Preserve causality by removing right-side steps introduced by padding
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.final_relu(out + residual)


class TCNTerrainRegression(nn.Module):
    """Terrain Regression Only TCN.

    Input:  x: [B, 200, 42]
    Output: terrain: [B, 30, 6]
    """

    def __init__(
        self,
        input_dim=42,
        hidden_channels=None,
        output_len=30,
        terrain_dim=6,
        kernel_size=3,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 128, 128]

        self.output_len = output_len
        self.terrain_dim = terrain_dim

        layers = []
        in_ch = input_dim

        # Stack causal dilated convolutions
        for i, out_ch in enumerate(hidden_channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.feature_dim = in_ch

        # Regression head maps historical features to terrain parameters for the next 30 steps
        self.terrain_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_len * terrain_dim),
        )

        self.apply(weights_init)

    def forward(self, x):
        """Input: x: [B, 200, 42]"""
        # 1. Clean invalid values
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Convert to Conv1d shape: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # 3. TCN extracts causal temporal features: [B, feature_dim, T]
        feature = self.encoder(x)

        # 4. Extract the final hidden state representing the full 200-frame causal history
        # [B, feature_dim, 200] -> [B, feature_dim]
        last_feature = feature[:, :, -1]

        # 5. Predict terrain for the next 30 steps [B, output_len, terrain_dim]
        terrain = self.terrain_head(last_feature)
        terrain = terrain.view(x.size(0), self.output_len, self.terrain_dim)

        return {"terrain": terrain}


# Compatibility aliases
TCNMultiTask = TCNTerrainRegression


if __name__ == "__main__":
    model = TCNTerrainRegression()
    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)
    print("TCN validation passed:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6)