import torch
import torch.nn as nn

from transformers import GPT2Config, GPT2Model


class RevIN(nn.Module):
    """Reversible Instance Normalization

    Mitigates distribution shifts across subjects, walking speeds, and IMU drift or bias.
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
    """GPT4TS multitask model (angle regression and gait classification)

    Architecture:
        RevIN -> Patch Embedding -> GPT-2 Backbone -> Temporal Projector
        -> {angle_head, gait_head}

    Input:
        x: [B, input_len, input_dim]   (default [B, 200, 42])
    Output:
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

        # 1. Compute fixed patch count: (200 - 16) // 8 + 1 = 24
        self.patch_num = (seq_len - patch_len) // stride + 1
        if self.patch_num <= 0:
            raise ValueError("Invalid patch_len / stride; no patches can be extracted")

        # 2. IMU multichannel instance normalization
        self.revin = RevIN(num_features=input_dim, affine=True)

        # 3. Patch feature projection [patch_len * input_dim] -> [d_model]
        self.patch_embedding = nn.Linear(input_dim * patch_len, d_model)

        # 4. Load the GPT-2 backbone
        try:
            config = GPT2Config.from_pretrained("gpt2")
        except Exception:
            # In offline environments, fall back to a randomly initialized equivalent configuration
            config = GPT2Config(n_embd=d_model, n_layer=12, n_head=12)
        config.output_attentions = False
        config.output_hidden_states = False
        config.use_cache = False

        # Force eager attention to avoid issues in certain PyTorch/CUDA combinations
        # where scaled_dot_product_attention can cause illegal memory access
        try:
            config._attn_implementation = "eager"
        except Exception:
            pass

        try:
            self.gpt2 = GPT2Model.from_pretrained("gpt2", config=config)
        except Exception:
            # In offline environments, use a randomly initialized equivalent GPT-2
            self.gpt2 = GPT2Model(config)

        # 4.5 Dimension consistency check
        # Patch embeddings and prediction heads use d_model, but the backbone output dimension
        # is determined by GPT-2 n_embd; mismatches otherwise surface only during forward,
        # so check them early.
        if d_model != self.gpt2.config.n_embd:
            raise ValueError(
                f"d_model ({d_model}) must equal GPT-2 n_embd "
                f"({self.gpt2.config.n_embd})，"
                f"otherwise the patch embedding output dimension differs from the backbone"
            )

        # 5. Fine-tune only LayerNorm and positional embeddings
        if freeze:
            for name, param in self.gpt2.named_parameters():
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # 6. Construct temporal projection in __init__ so the optimizer registers its parameters
        # Linearly map 24 patch tokens to 30 prediction steps
        self.temporal_projector = nn.Linear(
            self.patch_num, self.pred_len, bias=False
        )

        # 7. Angle regression head (eight outputs)
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, angle_dim),
        )

        # 8. Gait classification head (two-class logits)
        self.gait_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, gait_dim),
        )

    def forward(self, x, **kwargs):
        """Input:
            x: [B, input_len, input_dim]
        Output:
            {"angle": [B, pred_len, angle_dim],
             "gait":  [B, pred_len, gait_dim]}
        """
        if x.ndim != 3:
            raise ValueError("Input must be [B, T, F]")

        B, T, D = x.shape

        if T != self.seq_len:
            raise ValueError(
                f"Input length {T} differs from configured seq_len {self.seq_len}"
            )

        # The dataset has already applied nan_to_num and scaling by 1/100,
        # so do not call torch.nan_to_num on GPU tensors to avoid
        # illegal memory access with some PyTorch/CUDA versions.
        x = x.contiguous()

        # A. Instance normalization
        x = self.revin(x, mode="norm")

        # B. Split patches: [B, T, D] -> [B, patch_num, patch_len * D]
        patches = (
            x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(B, self.patch_num, self.patch_len * D)
        )

        # C. Project to token embeddings [B, patch_num, d_model]
        embeddings = self.patch_embedding(patches)

        # D. GPT-2 backbone attention processing
        # Note: GPT2Model automatically adds positional embeddings when inputs_embeds are passed
        # without explicit position_ids. Adding wpe manually as well previously resulted in
        # inputs + 2 * wpe; now only the backbone's own positional embedding is used.
        outputs = self.gpt2(inputs_embeds=embeddings)
        hidden_states = outputs.last_hidden_state  # [B, patch_num, d_model]

        # E. Temporal alignment
        # [B, patch_num, d_model] -> [B, d_model, patch_num]
        # -> [B, d_model, pred_len] -> [B, pred_len, d_model]
        h = hidden_states.transpose(1, 2)
        h = self.temporal_projector(h).transpose(1, 2)

        # F. Multitask outputs
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
    print("GPT4TS validation passed")
