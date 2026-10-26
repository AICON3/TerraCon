import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "Please install mamba-ssm: pip install causal-conv1d mamba-ssm"
    )

# =====================================================
# Mamba Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
#
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


class MambaBlock(nn.Module):
    """Standard Mamba block with residual connections and Pre-LayerNorm"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        pred_len=30,
        d_model=64,
        num_layers=2,
        terrain_dim=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.terrain_dim = terrain_dim

        # 1. Input temporal projection (42 -> d_model)
        self.embedding = nn.Linear(input_dim, d_model)

        # 2. Stacked Mamba backbone
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # 3. Normalization and regularization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. Terrain regression head (directly predicts [pred_len * 6] with channel ordering preserved)
        self.terrain_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * terrain_dim),
        )

        # 5. Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Input:

            x: [B, 200, 42] (IMU sequence)
        Output:
            dict: {"terrain": [B, 30, 6]}
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        B = x.size(0)

        # Input embedding [B, 200, d_model]
        h = self.embedding(x)

        # Pass through Mamba state-space layers
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        # Extract the final hidden state representing 200 frames of causal history [B, d_model]
        feature = h[:, -1, :]
        feature = self.dropout(feature)

        # Regress to [B, pred_len, terrain_dim]
        # Preserve channel order: [ipsi_x, ipsi_y, ipsi_z, contra_x, contra_y, contra_z]
        terrain = self.terrain_head(feature)
        terrain = terrain.view(B, self.pred_len, self.terrain_dim)

        return {"terrain": terrain}


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Arguments are compatible with build_model("mamba", cfg)
    model = MambaTerrainRegression(
        input_dim=42,
        pred_len=30,
        d_model=64,
    ).to(device)

    dummy_imu = torch.randn(8, 200, 42).to(device)
    out = model(dummy_imu)

    print("Mamba model test passed:")
    print("terrain shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])
    assert out["terrain"].shape == (8, 30, 6), "Output shape does not match the training task!"