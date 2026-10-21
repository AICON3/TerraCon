
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba=None

class MambaMultiTask(nn.Module):
    def __init__(self,input_dim=42,pred_len=30,d_model=64,angle_dim=8,gait_dim=2,**kwargs):
        super().__init__()
        self.pred_len=pred_len
        self.embedding=nn.Linear(input_dim,d_model)
        self.encoder=Mamba(d_model=d_model) if Mamba else nn.GRU(d_model,d_model,batch_first=True)
        self.angle_head=nn.Linear(d_model,angle_dim)
        self.gait_head=nn.Linear(d_model,gait_dim)
    def forward(self,x,**kwargs):
        x=self.embedding(x)
        y=self.encoder(x)
        x=y[0] if isinstance(y,tuple) else y
        x=x[:,-self.pred_len:,:]
        return {"angle":self.angle_head(x),"gait":self.gait_head(x)}
