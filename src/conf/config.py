from datetime import datetime

class cfg:
    rawdir = '../data/raw'
    prodir = '../data/processed'
    pthdir = '../data/model'
    logdir = '../data/logs'
    outdir = '../data/output'
    timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
    features = ['pm25','pm10_l1']
    target = ['pm25']
    knn = 2
    trnsize = .9
    nfolds = 2
    seed = 42
    idim = len(features)
    odim = len(target)
    hdim = 100
    wsize = 4
    bsize = 8
    loader = {'batch_size':bsize,
              'shuffle':False,
              'pin_memory':True,
              'drop_last':True}
    lr = 1e-4
    momentum = .5
    delta = 1.
    epochs = 4
