import os
from cfg import BaseConfig


class MainConfig(BaseConfig):
    def __init__(self, args=None):
        super(MainConfig, self).__init__(args=args)

        self.cfg.DATASET.ANNO_FILE=''
        self.cfg.DATASET.CLS_FILE = ''
        self.cfg.DATASET.N_SEGMENT = 8

        self.cfg.MODEL.FEAT_NET = 'resnet50'
        self.cfg.MODEL.PRIM_NET = 'mobilenetv2'
        self.cfg.MODEL.PRETRAINED = ''
        self.cfg.MODEL.FEAT_PRETRAINED = ''
        self.cfg.MODEL.POLICY_PRETRAINED = ''
        self.cfg.MODEL.CHANNEL_LIST = (0, 0.25, 0.5, 0.75, 1.0)

        self.cfg.TRAIN.STAGE = 'supernet'
        self.cfg.TRAIN.USE_EMA = False
        self.cfg.TRAIN.EMA_DECAY = 0.9
        self.cfg.TRAIN.SMOOTHING = 0.1
        self.cfg.TRAIN.RECALIBRATE_BN = False
        self.cfg.TRAIN.TEACHER_PRETRAINED = ''
        self.cfg.TRAIN.MARGIN = 0.3

        self.cfg.TRAIN.GATE_LR = 0.1
        self.cfg.TRAIN.POLICY_LR = 0.1
        self.cfg.TRAIN.CLS_FC_LR = 0.1
        self.cfg.TRAIN.USE_CONCAT = True
        self.cfg.TRAIN.USE_AUX_LOSS = True

        self.cfg.TRAIN.USE_META_NET = False
        self.cfg.TRAIN.META_LR = 1e-3
        self.cfg.TRAIN.META_WD = 1e-4
        self.cfg.TRAIN.META_NUM = 1000

        if args is not None:
            self.update_config(args)
