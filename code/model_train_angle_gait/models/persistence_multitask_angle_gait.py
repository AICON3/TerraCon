
import torch
import torch.nn as nn

class PersistenceMultiTask(nn.Module):
    def __init__(self,input_dim=42,pred_len=30,angle_dim=8,gait_dim=2,**kwargs):
        super().__init__()
        self.pred_len=pred_len
        self.angle=nn.Linear(input_dim,angle_dim)
        self.gait=nn.Linear(input_dim,gait_dim)
    def forward(self,x,**kwargs):
        h=x[:,-1,:]
        return {"angle":self.angle(h).unsqueeze(1).repeat(1,self.pred_len,1),
                "gait":self.gait(h).unsqueeze(1).repeat(1,self.pred_len,1)}
