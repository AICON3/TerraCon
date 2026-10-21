import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization)

    消除跨个体 (48人)、不同步速及 IMU 传感器温漂/偏置带来的分布偏移
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
            self.stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x
        return x


class GPT4TSTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        seq_len=200,
        pred_len=30,
        terrain_dim=6,
        patch_len=16,
        stride=8,
        d_model=768,
        freeze=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.terrain_dim = terrain_dim
        self.patch_len = patch_len
        self.stride = stride

        # 1. 计算固定的 Patch 数量: (200 - 16) // 8 + 1 = 24
        self.patch_num = (seq_len - patch_len) // stride + 1

        # 2. IMU 多通道实例归一化
        self.revin = RevIN(num_features=input_dim, affine=True)

        # 3. Patch 局部特征映射 [patch_len * input_dim] -> [d_model]
        self.patch_embedding = nn.Linear(input_dim * patch_len, d_model)

        # 4. GPT-2 骨干网络加载
        config = GPT2Config.from_pretrained("gpt2")
        config.output_attentions = False
        config.output_hidden_states = False
        self.gpt2 = GPT2Model.from_pretrained("gpt2", config=config)

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

        # 7. 地形参数回归头 (6维输出)
        self.terrain_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, terrain_dim),
        )

    def forward(self, x):
        """输入:

            x: [B, 200, 42]
        输出:
            {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 3:
            raise ValueError("Input must be [B, T, F]")

        B, T, D = x.shape

        # A. 实例归一化
        x = self.revin(x, mode="norm")

        # B. 切分 Patch: [B, T, D] -> [B, 24, 16 * 42]
        patches = (
            x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(B, self.patch_num, self.patch_len * D)
        )

        # C. 投射到 Token Embedding 空间 [B, 24, 768]
        embeddings = self.patch_embedding(patches)

        # D. 注入 GPT-2 绝对位置编码
        pos_ids = torch.arange(
            0, self.patch_num, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        embeddings = embeddings + self.gpt2.wpe(pos_ids)

        # E. GPT-2 骨干网络注意力交互
        outputs = self.gpt2(inputs_embeds=embeddings)
        hidden_states = outputs.last_hidden_state  # [B, 24, 768]

        # F. 时序跨度对齐 [B, 24, 768] -> [B, 768, 24] -> [B, 768, 30] -> [B, 30, 768]
        h = hidden_states.transpose(1, 2)
        h = self.temporal_projector(h).transpose(1, 2)

        # G. 回归地形目标参数 [B, 30, 6]
        terrain = self.terrain_head(h)

        return {"terrain": terrain}


if __name__ == "__main__":
    # 自检：实例化、参数注册检查以及张量传递
    model = GPT4TSTerrainRegression(input_dim=42, pred_len=30, terrain_dim=6)

    # 验证优化器可收集参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数张量数量: {len(trainable_params)}")

    # 验证 forward
    dummy_x = torch.randn(2, 200, 42)
    output = model(dummy_x)

    print("输出张量结构:", output["terrain"].shape)
    assert output["terrain"].shape == (2, 30, 6), "输出维度与实验预期不符！"
    print("模型验证通过，可直接投入训练！")