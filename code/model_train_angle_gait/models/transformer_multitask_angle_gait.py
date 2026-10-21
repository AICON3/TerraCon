
import torch
import torch.nn as nn

import math
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        self.pe=nn.Parameter(torch.zeros(1,max_len,d_model))
    def forward(self,x,**kwargs):
        return x+self.pe[:,:x.size(1)]

class TransformerMultiTask(nn.Module):
    def __init__(self,input_dim=42,d_model=128,nhead=4,num_layers=3,dim_feedforward=256,output_len=30,angle_dim=8,gait_dim=2,dropout=0.2,**kwargs):
        super().__init__()
        self.output_len=output_len
        self.proj=nn.Linear(input_dim,d_model)
        self.pos=PositionalEncoding(d_model)
        layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers)
        self.norm=nn.LayerNorm(d_model)
        self.angle_head=nn.Linear(d_model,angle_dim)
        self.gait_head=nn.Linear(d_model,gait_dim)
    def forward(self,x,**kwargs):
        x=self.norm(self.encoder(self.pos(self.proj(x))))
        x=x[:,-self.output_len:,:]
        return {"angle":self.angle_head(x),"gait":self.gait_head(x)}
