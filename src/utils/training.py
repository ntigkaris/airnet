import logging

import numpy as np

import torch
from torch.utils.data import DataLoader

from conf.config import cfg
from utils.model import AirnetDataset,Airnet
from utils.loss import AirnetLoss
from utils.metrics import rmse

logging.basicConfig(force=True,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(filename=f'{cfg.logdir}/LOG_{cfg.timestamp}.log'),
                              logging.StreamHandler()])

def trainFn(loader,
            model,
            criterion,
            optimizer,
            fold,
            epoch):
    model.train()
    lossTracker = []
    for i,(features,target) in enumerate(loader):
        prediction = model(features)
        loss = criterion(prediction,target)
        lossTracker.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(msg=f'AIRNET/TRAIN/fold={fold:02d}/epoch={epoch:03d}/batch={i}@{len(loader)}/loss: {loss:.2f}')
    logging.info(msg=f'AIRNET/TRAIN/fold={fold:02d}/epoch={epoch:03d}/batch=all/loss: {np.mean(lossTracker):.2f}')
    pass

def evalFn(loader,
           model,
           criterion,
           fold,
           epoch):
    model.eval()
    lossTracker = []
    predictions = []
    for i,(features,target) in enumerate(loader):
        with torch.no_grad():
            prediction = model(features)
            loss = criterion(prediction,target)
            lossTracker.append(loss.detach().numpy())
        predictions.append(prediction.numpy())
        logging.info(msg=f'AIRNET/EVAL/fold={fold:02d}/epoch={epoch:03d}/batch={i}@{len(loader)}/loss: {loss:.2f}')
    logging.info(msg=f'AIRNET/EVAL/fold={fold:02d}/epoch={epoch:03d}/batch=all/loss: {np.mean(lossTracker):.2f}')
    return np.concatenate(predictions)

def doTraining(df,fold):
    torch.manual_seed(seed=cfg.seed)
    trainFold = df[df['fold']!=fold]
    evalFold = df[df['fold']==fold]
    trainDataset = AirnetDataset(dataFrame=trainFold,windowSize=cfg.wsize)
    evalDataset = AirnetDataset(dataFrame=evalFold,windowSize=cfg.wsize)
    trainLoader = DataLoader(dataset=trainDataset,**cfg.loader)
    evalLoader = DataLoader(dataset=evalDataset,**cfg.loader)
    model = Airnet(inputDim=cfg.idim,
                   outputDim=cfg.odim,
                   hiddenDim=cfg.hdim,
                   windowSize=cfg.wsize,
                   batchSize=cfg.bsize)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=cfg.lr,
                                momentum=cfg.momentum)
    criterion = AirnetLoss(delta=cfg.delta)
    for epoch in range(cfg.epochs):
        trainFn(loader=trainLoader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                fold=fold,
                epoch=epoch)
        predictions = evalFn(loader=evalLoader,
                             model=model,
                             criterion=criterion,
                             fold=fold,
                             epoch=epoch)
    torch.save(obj={'model':model.state_dict(),
                    'predictions':predictions},
               f=f'{cfg.pthdir}/model_fold{fold:02d}_{cfg.timestamp}.pth')
    return rmse(v=np.array([i[1].detach().numpy() for i in evalDataset.data]),u=predictions)
