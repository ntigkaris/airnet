import torch
from torch.utils.data import Dataset

from conf.config import cfg

class AirnetDataset(Dataset):
    def __init__(self,
                 dataFrame,
                 windowSize):
        super(AirnetDataset,self).__init__()
        self.dataFrame = dataFrame
        self.windowSize = windowSize
        self.features = torch.tensor(data=dataFrame[cfg.features].values,dtype=torch.float32)
        self.target = torch.tensor(data=dataFrame[cfg.target].values,dtype=torch.float32)
        self.data = self.transformToSequences()
    def transformToSequences(self):
        dataset = []
        for i in range(len(self.features)-self.windowSize):
            sequence = self.features[i:i+self.windowSize]
            target = self.target[i+self.windowSize:i+self.windowSize+1]
            dataset.append((sequence,target))
        return dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]

class Airnet(torch.nn.Module):
    def __init__(self,
                 inputDim,
                 outputDim,
                 hiddenDim,
                 windowSize,
                 batchSize):
        super(Airnet,self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenDim = hiddenDim
        self.windowSize = windowSize
        self.batchSize = batchSize
        self.Wih = torch.nn.Parameter(data=torch.zeros(size=(hiddenDim*2,inputDim),requires_grad=True))
        self.Whh = torch.nn.Parameter(data=torch.zeros(size=(hiddenDim*2,hiddenDim),requires_grad=True))
        self.Bih = torch.nn.Parameter(data=torch.zeros(size=(hiddenDim*2,),requires_grad=True))
        self.Bhh = torch.nn.Parameter(data=torch.zeros(size=(hiddenDim*2,),requires_grad=True))
        self.head = torch.nn.Linear(in_features=hiddenDim,out_features=outputDim,bias=True)
    
    def forward(self,inputs):
        hiddenState = torch.zeros(size=(self.windowSize,self.hiddenDim),requires_grad=False)
        ret = torch.zeros(size=(self.batchSize,self.windowSize,self.hiddenDim),requires_grad=False)
        for i,x in enumerate(iterable=inputs):
            xf,xh = torch.nn.functional.linear(input=x,weight=self.Wih,bias=self.Bih).chunk(2,1)
            hf,hh = torch.nn.functional.linear(input=hiddenState,weight=self.Whh,bias=self.Bhh).chunk(2,1)
            fG = torch.sigmoid(input=xf+hf)
            hG = torch.tanh(input=xh+fG*hh)
            hiddenState = (1-fG)*hiddenState+fG*hG
            ret[i] = hiddenState
        return self.head(ret[:,-1,:])
