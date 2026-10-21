import torch
import torch.nn as nn

from transformers import GPT2Config, GPT2Model


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization)

    消除跨个体、不同步速及 IMU 传感器温漂/偏置带来的分布偏移。
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            diff = x - self.mean
            self.stdev = torch.sqrt(
                torch.mean(diff * diff, dim=1, keepdim=True) + self.eps
            ).detach()
            x = diff / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x
        return x


class GPT4TSMultiTask(nn.Module):
    """GPT4TS 多任务模型 (角度回归 + 步态分类)

    结构:
        RevIN -> Patch Embedding -> GPT-2 Backbone -> Temporal Projector
        -> {angle_head, gait_head}

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
        seq_len=200,
        patch_len=16,
        stride=8,
        d_model=768,
        freeze=True,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.angle_dim = angle_dim
        self.gait_dim = gait_dim
        self.patch_len = patch_len
        self.stride = stride

        # 1. 计算固定的 Patch 数量: (200 - 16) // 8 + 1 = 24
        self.patch_num = (seq_len - patch_len) // stride + 1
        if self.patch_num <= 0:
            raise ValueError("patch_len / stride 不合法，无法切出任何 patch")

        # 2. IMU 多通道实例归一化
        self.revin = RevIN(num_features=input_dim, affine=True)

        # 3. Patch 局部特征映射 [patch_len * input_dim] -> [d_model]
        self.patch_embedding = nn.Linear(input_dim * patch_len, d_model)

        # 4. GPT-2 骨干网络加载
        try:
            config = GPT2Config.from_pretrained("gpt2")
        except Exception:
            # 离线环境下退化为同结构随机初始化配置
            config = GPT2Config(n_embd=d_model, n_layer=12, n_head=12)
        config.output_attentions = False
        config.output_hidden_states = False
        config.use_cache = False

        # 强制使用 eager attention，避免某些 PyTorch/CUDA 组合下
        # scaled_dot_product_attention 导致 illegal memory access
        try:
            config._attn_implementation = "eager"
        except Exception:
            pass

        try:
            self.gpt2 = GPT2Model.from_pretrained("gpt2", config=config)
        except Exception:
            # 离线环境下退化为同结构随机初始化 GPT-2
            self.gpt2 = GPT2Model(config)

        # 4.5 维度一致性校验
        # patch_embedding / 预测头都按 d_model 构造，而主干输出维度由
        # GPT-2 的 n_embd 决定，两者不一致会在 forward 里才报错，
        # 这里提前拦住。
        if d_model != self.gpt2.config.n_embd:
            raise ValueError(
                f"d_model ({d_model}) 必须等于 GPT-2 的 n_embd "
                f"({self.gpt2.config.n_embd})，"
                f"否则 patch embedding 输出维度与主干不匹配"
            )

        # 5. 参数微调策略 (仅开启 LayerNorm 和 Position Embedding 更新)
        if freeze:
            for name, param in self.gpt2.named_parameters():
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # 6. 时序跨度投影: 在 __init__ 中静态构建，保证被优化器完全收录
        # 将 Patch 序列长度 (24) 线性映射到目标预测步长 (30)
        self.temporal_projector = nn.Linear(
            self.patch_num, self.pred_len, bias=False
        )

        # 7. 角度回归头 (8 维输出)
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, angle_dim),
        )

        # 8. 步态分类头 (2 类 logits)
        self.gait_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, gait_dim),
        )

    def forward(self, x, **kwargs):
        """输入:
            x: [B, input_len, input_dim]
        输出:
            {"angle": [B, pred_len, angle_dim],
             "gait":  [B, pred_len, gait_dim]}
        """
        if x.ndim != 3:
            raise ValueError("Input must be [B, T, F]")

        B, T, D = x.shape

        if T != self.seq_len:
            raise ValueError(
                f"输入序列长度 {T} 与模型设定 seq_len {self.seq_len} 不一致"
            )

        # 数据已在 dataset 中完成 nan_to_num 和 /100 归一化，
        # 这里不再对 GPU 张量调用 torch.nan_to_num，避免部分
        # PyTorch/CUDA 版本上的非法内存访问风险。
        x = x.contiguous()

        # A. 实例归一化
        x = self.revin(x, mode="norm")

        # B. 切分 Patch: [B, T, D] -> [B, patch_num, patch_len * D]
        patches = (
            x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(B, self.patch_num, self.patch_len * D)
        )

        # C. 投射到 Token Embedding 空间 [B, patch_num, d_model]
        embeddings = self.patch_embedding(patches)

        # D. GPT-2 骨干网络注意力交互
        # 注意: GPT2Model 在收到 inputs_embeds 且未显式传入 position_ids 时,
        # 内部会自动加上一层 wpe 位置编码。此前这里又手工加了一次, 相当于
        # inputs + 2 * wpe, 位置编码被叠加两次。此处只保留主干内部那一次。
        outputs = self.gpt2(inputs_embeds=embeddings)
        hidden_states = outputs.last_hidden_state  # [B, patch_num, d_model]

        # E. 时序跨度对齐
        # [B, patch_num, d_model] -> [B, d_model, patch_num]
        # -> [B, d_model, pred_len] -> [B, pred_len, d_model]
        h = hidden_states.transpose(1, 2)
        h = self.temporal_projector(h).transpose(1, 2)

        # F. 多任务输出
        angle = self.angle_head(h)  # [B, pred_len, angle_dim]
        gait = self.gait_head(h)    # [B, pred_len, gait_dim]

        return {"angle": angle, "gait": gait}


if __name__ == "__main__":
    model = GPT4TSMultiTask(input_dim=42, seq_len=200, pred_len=30, freeze=True)
    dummy_x = torch.randn(2, 200, 42)
    output = model(dummy_x)
    print("angle:", output["angle"].shape)
    print("gait:", output["gait"].shape)
    assert output["angle"].shape == (2, 30, 8)
    assert output["gait"].shape == (2, 30, 2)
    print("GPT4TS 验证通过")
