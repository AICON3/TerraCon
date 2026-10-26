import torch
import torch.nn as nn
import torch.nn.functional as F


def series_decomp(x, kernel_size=25):
    """序列分解: 用移动平均把序列拆成 seasonal + trend

    x: [B, T, d_model]

    返回:
        seasonal: 高频残差部分, 形状同 x
        trend:    移动平均得到的低频趋势部分, 形状同 x

    kernel_size 会被裁剪到不超过序列长度并保持为奇数，保证
    avg_pool1d 的输出长度与输入一致。
    """
    T = x.size(1)

    k = min(int(kernel_size), T)
    if k % 2 == 0:
        k -= 1
    k = max(k, 1)

    if k == 1:
        # Skip decomposition when the sequence is too short
        return x, torch.zeros_like(x)

    padding = (k - 1) // 2

    trend = F.avg_pool1d(
        x.transpose(1, 2),
        kernel_size=k,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ).transpose(1, 2)

    trend = trend[:, :T, :]

    return x - trend, trend


class TimeMixing(nn.Module):
    """沿时间轴的 token mixing + 特征维通道混合

    真实 TimeMixer 的 mixing 作用在时间维度上(不同时刻相互交互)。
    这里用共享权重的深度可分离卷积沿时间轴耦合各时刻(参数量与
    序列长度无关)，再接一层特征维 MLP，最后带残差。
    """

    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Temporal coupling: depthwise convolution applies independently to each channel
        self.time_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        # Channel mixing over feature dimensions
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        h = self.norm(x)
        h = self.time_conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.channel_mix(h)
        return x + self.dropout(h)


class PastDecomposableMixing(nn.Module):
    """Past Decomposable Mixing (TimeMixer 核心单元)

    对多个尺度的序列:
        1. 每个尺度做序列分解, 得到 seasonal / trend
        2. Bottom-up: 细尺度 trend 下采样后汇入粗尺度 trend
        3. Top-down:  粗尺度 seasonal 上采样后汇入细尺度 seasonal
        4. 各尺度的 seasonal / trend 分别做时间轴 mixing

    输入 / 输出均为 list, 每个元素形如 [B, T_scale, d_model]。
    """

    def __init__(
        self,
        d_model,
        down_sampling_window=2,
        moving_avg=25,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        self.down_sampling_window = down_sampling_window
        self.moving_avg = moving_avg

        self.seasonal_mixing = TimeMixing(d_model, kernel_size, dropout)
        self.trend_mixing = TimeMixing(d_model, kernel_size, dropout)

    def forward(self, x_list):
        num_scales = len(x_list)

        # 1) Decompose each scale separately
        seasonal_list = []
        trend_list = []
        for x in x_list:
            seasonal, trend = series_decomp(x, self.moving_avg)
            seasonal_list.append(seasonal)
            trend_list.append(trend)

        # 2) Bottom-up: downsample fine-scale trends and merge into coarse-scale trends
        for i in range(num_scales - 1):
            down = F.avg_pool1d(
                trend_list[i].transpose(1, 2),
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window,
            ).transpose(1, 2)

            if down.size(1) != trend_list[i + 1].size(1):
                down = F.interpolate(
                    down.transpose(1, 2),
                    size=trend_list[i + 1].size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            trend_list[i + 1] = trend_list[i + 1] + down

        # 3) Top-down: upsample coarse-scale seasonality and merge into fine-scale seasonality
        for i in range(num_scales - 1, 0, -1):
            up = F.interpolate(
                seasonal_list[i].transpose(1, 2),
                size=seasonal_list[i - 1].size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

            seasonal_list[i - 1] = seasonal_list[i - 1] + up

        # 4) Apply temporal mixing to seasonal and trend components at each scale
        out = []
        for seasonal, trend in zip(seasonal_list, trend_list):
            out.append(
                self.seasonal_mixing(seasonal) + self.trend_mixing(trend)
            )

        return out


class TimeMixerMultiTask(nn.Module):
    """TimeMixer 多任务模型 (角度回归 + 步态分类)

    结构:
        多尺度下采样 -> 各尺度投影
        -> PastDecomposableMixing x num_pdm_layers
           (序列分解 + Bottom-up/Top-down 跨尺度混合 + 时间轴 mixing)
        -> 跨尺度特征融合 -> {angle_head, gait_head}

    输入:
        x: [B, input_len, input_dim]   (默认 [B, 200, 42])
    输出:
        {"angle": [B, pred_len, angle_dim],
         "gait":  [B, pred_len, gait_dim]}
    """

    def __init__(
        self,
        input_dim=42,
        pred_len=30,
        angle_dim=8,
        gait_dim=2,
        d_model=64,
        num_scales=3,
        down_sampling_window=2,
        moving_avg=25,
        num_pdm_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.angle_dim = angle_dim
        self.gait_dim = gait_dim
        self.num_scales = num_scales
        self.down_sampling_window = down_sampling_window
        self.moving_avg = moving_avg

        # 1. Input projection at each scale (input_dim -> d_model)
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, d_model) for _ in range(num_scales)
        ])

        # 2. Past Decomposable Mixing blocks across all scales
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                d_model=d_model,
                down_sampling_window=down_sampling_window,
                moving_avg=moving_avg,
                dropout=dropout,
            )
            for _ in range(num_pdm_layers)
        ])

        # 3. Multiscale fusion by concatenating pooled features
        total_dim = d_model * num_scales
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.GELU(),
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
        )

        # 4. Angle regression head [B, total_dim] -> [B, pred_len, angle_dim]
        self.angle_head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * angle_dim),
        )

        # 5. Gait classification head [B, total_dim] -> [B, pred_len, gait_dim]
        self.gait_head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * gait_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        """输入:
            x: [B, input_len, input_dim]
        输出:
            {"angle": [B, pred_len, angle_dim],
             "gait":  [B, pred_len, gait_dim]}
        """
        if x.ndim != 3:
            raise ValueError("Input must be [B, T, F]")

        # The dataset has already applied nan_to_num and scaling by 1/100,
        # so do not call torch.nan_to_num on GPU tensors to avoid
        # illegal memory access with some PyTorch/CUDA versions.
        x = x.contiguous()

        # Dimension consistency check
        if x.size(-1) != self.input_dim:
            if x.size(1) == self.input_dim:
                x = x.transpose(1, 2).contiguous()
            else:
                raise ValueError(
                    f"输入特征维度 {x.shape} 与设定 input_dim {self.input_dim} 不一致"
                )

        B = x.size(0)

        # Multiscale downsampling and projection to [B, T_level, d_model]
        scale_features = []
        current = x

        for level in range(self.num_scales):
            scale_features.append(self.encoders[level](current))

            # Downsample to a coarser temporal sequence:
            # [B, T, F] -> [B, F, T] -> avg_pool -> [B, T_new, F]
            if level < self.num_scales - 1:
                current = F.avg_pool1d(
                    current.transpose(1, 2),
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                ).transpose(1, 2)

        # Past Decomposable Mixing:
        # Decompose each scale -> Bottom-up/Top-down mixing -> temporal mixing
        for pdm in self.pdm_blocks:
            scale_features = pdm(scale_features)

        # Globally pool and concatenate temporal features across scales
        pooled = [h.mean(dim=1) for h in scale_features]

        # Cross-scale fusion [B, d_model * num_scales]
        multi_scale_feature = torch.cat(pooled, dim=-1)
        feature = self.fusion(multi_scale_feature)

        # Multitask outputs reshaped to [B, pred_len, dim]
        angle = self.angle_head(feature)
        angle = angle.view(B, self.pred_len, self.angle_dim)

        gait = self.gait_head(feature)
        gait = gait.view(B, self.pred_len, self.gait_dim)

        return {"angle": angle, "gait": gait}


if __name__ == "__main__":
    model = TimeMixerMultiTask(input_dim=42, pred_len=30, d_model=64)
    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)
    print("angle:", out["angle"].shape)
    print("gait:", out["gait"].shape)
    assert out["angle"].shape == (8, 30, 8)
    assert out["gait"].shape == (8, 30, 2)
    print("TimeMixer 验证通过")
