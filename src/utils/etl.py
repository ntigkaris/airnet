import os
import re

import numpy as np
import pandas as pd

from conf.config import cfg

class DataObject(object):
    def __init__(self,filename):
        super(DataObject,self).__init__()
        self.filename = filename
        self.xlsxParams = {'io':f'{cfg.rawdir}/{self.filename}',
                           'sheet_name':None,
                           'skiprows':0,
                           'header':0}
        self.sheetname = 'Στ. ΕΓΝΑΤΙΑΣ'
        self.keepColumns = ['SO2\nμg/m3','PM10\nμg/m3','PM2,5\nμg/m3','CO\nmg/m3','NO\nμg/m3','NO2\nμg/m3','O3\nμg/m3','Θερμοκ -\nρασία\no C','Σχετική\nΥγρασία\n%']
        self.columnNames = ['so2','pm10','pm25','co','no','no2','o3','temp','rh']

def extract(obj):
    dataFrame = pd.read_excel(**obj.xlsxParams)
    return dataFrame[obj.sheetname].filter(items=obj.keepColumns,axis=1)\
                                   .set_axis(labels=obj.columnNames,axis=1)

def integrate():
    rawData = [extract(obj=DataObject(f)) for f in os.listdir(path=cfg.rawdir) if f.startswith('poll')]
    return pd.concat(objs=rawData,axis=0)\
             .replace(to_replace=r'^\s*$',value=np.nan,regex=True)\
             .dropna(how='all')

def write(dataFrame):
    dataFrame.to_csv(f"{cfg.prodir}/poll_{'_'.join([re.sub(r'(?:poll_|.xlsx)','',f) for f in os.listdir(cfg.rawdir)])}.csv",mode='w',index=False)
    pass

def doEtl():
    write(dataFrame=integrate())
    pass
