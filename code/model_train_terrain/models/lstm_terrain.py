import torch
import torch.nn as nn

# =====================================================
# LSTM Terrain Regression Only
#
# Input:
#   x: [B, 200, 42]
#
# Output:
#   dict: {"terrain": [B, 30, 6]}
# =====================================================


class LSTMTerrainRegression(nn.Module):

    def __init__(
        self,
        input_dim=42,
        hidden_dim=128,
        num_layers=2,
        pred_len=30,
        terrain_dim=6,
        dropout=0.2,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.terrain_dim = terrain_dim

        # 1. Base temporal encoder: bidirectional or multilayer LSTM (dropout if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. Hidden-state normalization and regularization
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. Map the final LSTM hidden state to [pred_len, terrain_dim]
        # Avoid biases introduced by selecting only a trailing slice
        self.terrain_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * terrain_dim),
        )

        # 4. Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def forward(self, x):
        """Input:

            x: [B, 200, 42] (IMU sequence)
        Output:
            dict: {"terrain": [B, 30, 6]}
        """
        # Clean invalid values
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim != 3:
            raise ValueError(f"Input must be [B, T, F], but got {x.shape}")

        B = x.size(0)

        # LSTM temporal features: out [B, T, hidden_dim]
        out, _ = self.lstm(x)

        # Take the final hidden state representing the full 200-frame history [B, hidden_dim]
        last_hidden = out[:, -1, :]
        last_hidden = self.norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Project to [B, pred_len * terrain_dim] and reshape to [B, pred_len, terrain_dim]
        terrain = self.terrain_head(last_hidden)
        terrain = terrain.view(B, self.pred_len, self.terrain_dim)

        return {"terrain": terrain}


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    # Arguments match build_model("lstm", cfg)
    model = LSTMTerrainRegression(
        input_dim=42,
        hidden_dim=128,
        num_layers=2,
        pred_len=30,
        terrain_dim=6,
        dropout=0.2,
    )

    x = torch.randn(8, 200, 42)
    out = model(x)

    print("LSTM validation passed:")
    print("terrain output shape:", out["terrain"].shape)  # torch.Size([8, 30, 6])