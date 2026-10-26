import math
import torch
import torch.nn as nn

# =====================================================
# Transformer Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class TransformerTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        output_len=30,
        terrain_dim=6,
        dropout=0.2,
    ):
        super().__init__()
        self.output_len = output_len
        self.terrain_dim = terrain_dim

        # 1. Linear projection and positional encoding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # 2. Transformer Encoder backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 3. Project the complete 200-frame history to 30 future steps
        # Match baseline models to avoid losing historical context through simple truncation
        self.time_projector = nn.Linear(200, output_len)

        # 4. Six-channel terrain regression head with channel alignment preserved
        self.terrain_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, terrain_dim),
        )

        self.apply(weights_init)

    def forward(self, x):
        """输入:

            x: [B, 200, 42]
        输出:
            dict: {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        # A. Project and add positional encodings
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        # B. Transformer attention encoding
        h = self.encoder(h)
        h = self.norm(h)
        h = self.dropout(h)  # [B, 200, d_model]

        # C. Temporal projection to output_len (30 steps)
        # [B, 200, d_model] -> [B, d_model, 200] -> [B, d_model, 30] -> [B, 30, d_model]
        h = self.time_projector(h.transpose(1, 2)).transpose(1, 2)

        # D. Regress terrain parameters [B, 30, 6]
        # Channel order matches the dataset: [ipsi_x, ipsi_y, ipsi_z, contra_x, contra_y, contra_z]
        terrain = self.terrain_head(h)

        return {"terrain": terrain}


# Compatibility aliases
TransformerMultiTask = TransformerTerrainRegression


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    # Arguments match build_model("transformer", cfg)
    model = TransformerTerrainRegression(
        input_dim=42,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        output_len=30,
        dropout=0.2,
    )

    dummy_x = torch.randn(8, 200, 42)
    out = model(dummy_x)

    print("Transformer 验证通过:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "输出维度与训练任务要求不符！"