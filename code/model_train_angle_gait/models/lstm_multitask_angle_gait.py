
import torch
import torch.nn as nn

class LSTMMultiTask(nn.Module):
    def __init__(self,input_dim=42,hidden_dim=128,num_layers=2,pred_len=30,angle_dim=8,gait_dim=2,dropout=0.2,**kwargs):
        super().__init__()
        self.pred_len=pred_len
        self.lstm=nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True,dropout=dropout)
        self.norm=nn.LayerNorm(hidden_dim)
        self.angle_head=nn.Linear(hidden_dim,angle_dim)
        self.gait_head=nn.Linear(hidden_dim,gait_dim)
    def forward(self,x,**kwargs):
        h,_=self.lstm(x)
        h=self.norm(h[:,-self.pred_len:,:])
        return {"angle":self.angle_head(h),"gait":self.gait_head(h)}
