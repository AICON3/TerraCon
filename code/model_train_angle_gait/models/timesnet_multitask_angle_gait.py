import torch
import torch.nn as nn
import torch.nn.functional as F


def FFT_for_Period(x, k=5):
    """提取时序主要周期及其对应的频率幅值权重

    x: [B, T, C]

    返回:
        period:        list[int]，长度 k，各主要分量的周期长度
        period_weight: [B, k]，各周期对应频率的幅值，
                       供后续 softmax 自适应加权融合使用
    """
    xf = torch.fft.rfft(x, dim=1)
    amplitude = torch.abs(xf).mean(dim=-1)      # [B, T//2 + 1]

    frequency = amplitude.mean(dim=0)           # [T//2 + 1]
    # 用 clone 避免在 autograd 图上做原地修改
    frequency = frequency.clone()
    frequency[0] = 0.0                          # 压制直流分量

    # 保证 k 不超过频域采样点总数
    k = max(min(k, frequency.size(0) - 1), 1)
    _, top_list = torch.topk(frequency, k)

    period = []
    for top in top_list.detach().cpu().tolist():
        # 防御除零风险
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
                # 用 new_zeros 保持 dtype / device 与输入一致
                # (torch.zeros 默认 float32，混合精度下会导致 cat 报错)
                padding = x.new_zeros(B, pad_len, C)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x

            # 变换到 2D 周期空间: [B, length, C] -> [B, C, length//period, period]
            out = out.reshape(B, length // period, period, C).permute(
                0, 3, 1, 2
            )
            out = self.conv(out)
            # 还原为 1D 时序: [B, C, length//period, period] -> [B, length, C]
            out = out.permute(0, 2, 3, 1).reshape(B, length, C)
            out = out[:, :T, :]
            res.append(out)

        # 多周期自适应加权融合
        # 按各周期对应频率的幅值做 softmax 加权，而不是简单取平均
        res = torch.stack(res, dim=-1)                      # [B, T, C, k]
        period_weight = F.softmax(
            period_weight.float(), dim=1
        )                                                   # [B, k]
        period_weight = period_weight.to(res.dtype).unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * period_weight, dim=-1)        # [B, T, C]
        return res + x


class TimesNetMultiTask(nn.Module):
    """TimesNet 多任务模型 (角度回归 + 步态分类)

    结构:
        Linear Embedding -> TimesBlock x num_layers -> LayerNorm
        -> 时序长度投影 -> {angle_head, gait_head}

    输入:
        x: [B, input_len, input_dim]   (默认 [B, 200, 42])
    输出:
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

        # 3. 时序长度投影: 将完整的 input_len 帧聚合映射到目标 pred_len
        self.predict_linear = nn.Linear(input_len, pred_len)

        # 4. 角度回归头 (angle_dim 维输出)
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, angle_dim),
        )

        # 5. 步态分类头 (gait_dim 类 logits)
        self.gait_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, gait_dim),
        )

    def forward(self, x, **kwargs):
        """输入:
            x: [B, input_len, input_dim]
        输出:
            {"angle": [B, pred_len, angle_dim],
             "gait":  [B, pred_len, gait_dim]}
        """
        # 数据已在 dataset 中完成 nan_to_num 和 /100 归一化，
        # 这里不再对 GPU 张量调用 torch.nan_to_num，避免部分
        # PyTorch/CUDA 版本上的非法内存访问风险。
        x = x.contiguous()
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        B, T, _ = x.shape

        if T != self.input_len:
            raise ValueError(
                f"输入序列长度 {T} 与模型设定 input_len {self.input_len} 不一致"
            )

        # A. 线性嵌入
        h = self.embedding(x)
        h = self.dropout(h)

        # B. 周期性 2D 卷积转换
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)  # [B, input_len, d_model]

        # C. 跨时序投影到预测步长
        # [B, input_len, d_model] -> [B, d_model, input_len]
        # -> [B, d_model, pred_len] -> [B, pred_len, d_model]
        h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)

        # D. 多任务输出
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
    print("TimesNet 验证通过")
