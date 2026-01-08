import torch

class AirnetLoss(torch.nn.Module):
    def __init__(self,delta):
        super(AirnetLoss,self).__init__()
        self.delta = delta
    def forward(self,X,y):
        z = torch.abs(input=X-y)
        return torch.where(condition=z<=self.delta,
                           input=.5*z**2,
                           other=self.delta*(z-.5*self.delta))\
                    .mean()
