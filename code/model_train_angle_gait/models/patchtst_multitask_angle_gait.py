import torch
import torch.nn as nn
import torch.nn.functional as F



# ======================================================
# Position Encoding
# ======================================================

class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model):

        super().__init__()

        self.pe = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )


    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]



# ======================================================
# Patch Embedding
# ======================================================

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        patch_len,
        stride,
        in_channels,
        d_model
    ):

        super().__init__()

        self.patch_len = patch_len
        self.stride = stride


        self.proj = nn.Linear(
            patch_len * in_channels,
            d_model
        )


    def forward(self,x):

        # x:
        # [B,T,C]

        B,T,C = x.shape


        x = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride
        )


        # [B,num_patch,C,patch_len]

        x = x.reshape(
            B,
            x.size(1),
            C*self.patch_len
        )


        return self.proj(x)





# ======================================================
# Channel Independent Processor
# ======================================================

class ChannelIndependentProcessor(nn.Module):

    """
    Channel Independent PatchTST

    输入:
        [B,T,C]

    每个channel独立PatchTST编码

    输出:
        [B,C,d_model]

    """


    def __init__(
        self,
        input_len,
        patch_len,
        stride,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward,
        dropout
    ):

        super().__init__()


        num_patches = (
            input_len - patch_len
        ) // stride + 1


        self.num_patches = num_patches



        self.patch_embed = PatchEmbedding(
            patch_len,
            stride,
            in_channels=1,
            d_model=d_model
        )


        self.pos_enc = PositionalEncoding(
            num_patches,
            d_model
        )



        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=n_heads,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            activation="gelu",

            batch_first=True

        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=n_layers

        )


        self.norm = nn.LayerNorm(
            d_model
        )




    def forward(self,x):

        """

        x:

        [B,T,C]


        """

        B,T,C = x.shape



        # 每个传感器独立处理

        x = x.permute(
            0,
            2,
            1
        )


        # [B*C,T,1]

        x = x.reshape(
            B*C,
            T,
            1
        )



        x = self.patch_embed(x)


        x = self.pos_enc(x)


        x = self.encoder(x)


        x = self.norm(x)



        # [B,C,num_patch,d]

        x = x.reshape(
            B,
            C,
            self.num_patches,
            -1
        )


        # patch平均

        x = x.mean(
            dim=2
        )


        return x






# ======================================================
# PatchTST Multi Task
# ======================================================

class PatchTSTMultiTask(nn.Module):


    def __init__(

        self,

        input_len=200,

        pred_len=30,

        input_dim=42,

        angle_dim=8,

        gait_dim=2,

        d_model=128,

        n_heads=8,

        n_layers=3,

        dim_feedforward=512,

        patch_len=16,

        stride=8,

        dropout=0.1,

        channel_independent=True,

        **kwargs

    ):


        super().__init__()



        self.input_len=input_len

        self.pred_len=pred_len

        self.input_dim=input_dim

        self.angle_dim=angle_dim

        self.gait_dim=gait_dim



        self.channel_independent=True



        # ==================================================
        # CI PatchTST backbone
        # ==================================================

        self.processor = ChannelIndependentProcessor(

            input_len=input_len,

            patch_len=patch_len,

            stride=stride,

            d_model=d_model,

            n_heads=n_heads,

            n_layers=n_layers,

            dim_feedforward=dim_feedforward,

            dropout=dropout

        )




        # ==================================================
        # Prediction feature
        # ==================================================

        self.fc = nn.Linear(
            d_model,
            d_model
        )



        # ==================================================
        # Heads
        # ==================================================

        self.angle_head = nn.Linear(

            d_model,

            pred_len*angle_dim

        )


        self.gait_head = nn.Linear(

            d_model,

            pred_len*gait_dim

        )



        self._init_weights()




    def _init_weights(self):

        for m in self.modules():

            if isinstance(m,nn.Linear):

                nn.init.xavier_uniform_(

                    m.weight,

                    gain=0.5

                )


                if m.bias is not None:

                    nn.init.zeros_(m.bias)




    def forward(self,x):


        """

        x:

        [B,200,42]

        """


        B,T,C = x.shape



        if T != self.input_len:

            raise ValueError(
                f"T mismatch {T}"
            )


        if C != self.input_dim:

            raise ValueError(
                f"C mismatch {C}"
            )



        # ==========================
        # CI encoder
        # ==========================

        features = self.processor(x)


        # channel fusion

        feature = features.mean(
            dim=1
        )


        feature = F.relu(
            self.fc(feature)
        )



        # ==========================
        # Angle
        # ==========================

        angle = self.angle_head(
            feature
        )


        angle = angle.view(
            B,
            self.pred_len,
            self.angle_dim
        )



        # ==========================
        # Gait
        # ==========================

        gait = self.gait_head(
            feature
        )


        gait = gait.view(
            B,
            self.pred_len,
            self.gait_dim
        )



        return {

            "angle":angle,

            "gait":gait

        }




    def freeze_backbone(self):

        for name,param in self.named_parameters():

            if "processor" in name:

                param.requires_grad=False




    def unfreeze_all(self):

        for param in self.parameters():

            param.requires_grad=True





# ======================================================
# Test
# ======================================================

if __name__=="__main__":


    model = PatchTSTMultiTask()


    x=torch.randn(
        4,
        200,
        42
    )


    out=model(x)


    print(
        out["angle"].shape
    )


    print(
        out["gait"].shape
    )