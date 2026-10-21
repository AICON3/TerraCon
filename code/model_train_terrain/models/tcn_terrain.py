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

        # 因果卷积分支 1
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

        # 因果卷积分支 2
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

        # 残差对齐
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = x

        # 保证因果性：切掉右侧由 padding 产生的未来步
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

        # 堆叠因果膨胀卷积
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

        # 回归头：从历史汇聚特征映射到未来 30 步的地形参数
        self.terrain_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_len * terrain_dim),
        )

        self.apply(weights_init)

    def forward(self, x):
        """输入: x: [B, 200, 42]"""
        # 1. 异常值清理
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. 变换为 Conv1d 格式: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # 3. TCN 提取时序因果特征: [B, feature_dim, T]
        feature = self.encoder(x)

        # 4. 提取第 200 帧末尾隐状态（凝聚了 200 帧所有因果历史信息）
        # [B, feature_dim, 200] -> [B, feature_dim]
        last_feature = feature[:, :, -1]

        # 5. 投影到未来 30 步的地形 [B, output_len, terrain_dim]
        terrain = self.terrain_head(last_feature)
        terrain = terrain.view(x.size(0), self.output_len, self.terrain_dim)

        return {"terrain": terrain}


# 兼容别名
TCNMultiTask = TCNTerrainRegression


if __name__ == "__main__":
    model = TCNTerrainRegression()
    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)
    print("TCN 验证成功:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6)