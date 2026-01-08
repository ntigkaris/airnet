import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,KFold

from conf.config import cfg
from utils.impute import AirnetImputer

def doPreprocessing():
    processedData = [pd.read_csv(filepath_or_buffer=f'{cfg.prodir}/{f}') for f in os.listdir(path=cfg.prodir) if f.startswith('poll')]
    dataFrame = pd.concat(objs=processedData,axis=0)
    dataFrame = AirnetImputer(numNeighbors=cfg.knn).fit_transform(dataFrame)
    dataFrame['pm10_l1'] = dataFrame.pm10.shift(periods=1)
    trainData,testData = train_test_split(dataFrame,train_size=cfg.trnsize,shuffle=False)
    Fold = KFold(n_splits=cfg.nfolds,shuffle=False)
    for i,(_,evalIndex) in enumerate(Fold.split(trainData)):
        trainData.loc[evalIndex,'fold'] = i
    trainData['fold'] = trainData['fold'].astype(np.uint8)
    return trainData,testData
