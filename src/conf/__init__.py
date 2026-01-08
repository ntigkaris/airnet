import os

from conf.config import cfg

os.makedirs(name=cfg.logdir,exist_ok=True)
os.makedirs(name=cfg.prodir,exist_ok=True)
os.makedirs(name=cfg.pthdir,exist_ok=True)
os.makedirs(name=cfg.outdir,exist_ok=True)
