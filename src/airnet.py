import logging

import numpy as np

from conf.config import cfg
from utils.etl import doEtl
from utils.preprocessing import doPreprocessing
from utils.training import doTraining

logging.basicConfig(force=True,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(filename=f'{cfg.logdir}/LOG_{cfg.timestamp}.log'),
                              logging.StreamHandler()])

if __name__=='__main__':
    doEtl()
    trainData,testData = doPreprocessing()
    trainScore =  np.mean(a=[doTraining(trainData,fold) for fold in range(cfg.nfolds)])
    logging.info(msg=f'AIRNET/TRAIN/fold=all/score: {trainScore:.2f}')
