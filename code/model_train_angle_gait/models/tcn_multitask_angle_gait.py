
import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self,in_c,out_c,kernel=3,dilation=1,dropout=0.2):
        super().__init__()
        pad=(kernel-1)*dilation
        self.net=nn.Sequential(
            nn.Conv1d(in_c,out_c,kernel,padding=pad,dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_c,out_c,kernel,padding=pad,dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.down=nn.Conv1d(in_c,out_c,1) if in_c!=out_c else nn.Identity()
    def forward(self,x):
        y=self.net(x)
        y=y[:,:,:x.size(2)]
        return torch.relu(y+self.down(x))

class TCNMultiTask(nn.Module):
    def __init__(self,input_dim=42,hidden_channels=[64,128,128],output_len=30,angle_dim=8,gait_dim=2,dropout=0.2):
        super().__init__()
        layers=[]; c=input_dim
        for i,h in enumerate(hidden_channels):
            layers.append(TemporalBlock(c,h,dilation=2**i,dropout=dropout)); c=h
        self.encoder=nn.Sequential(*layers)
        self.angle_head=nn.Linear(c,angle_dim)
        self.gait_head=nn.Linear(c,gait_dim)
        self.output_len=output_len
    def forward(self,x,**kwargs):
        x=self.encoder(x.transpose(1,2)).transpose(1,2)
        x=x[:,-self.output_len:,:]
        return {"angle":self.angle_head(x),"gait":self.gait_head(x)}
