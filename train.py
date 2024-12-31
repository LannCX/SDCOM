import os
import random
import pprint
import argparse
import warnings

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

import init_path
from utils.logger import Logger
from data.data_loader import customer_data_loader
from arch.net_loader import customer_net_loader

# **************************************************************************
# Replace by your custom class
# **************************************************************************
from trainer.main_trainer import MainTrainer as Trainer
from cfg.main_cfg import MainConfig as Config
# **************************************************************************


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', required=False, type=str, default='./exp_configs/anet-policy-local.yaml', 
                        help='experiment configure file name')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, 
                        help="Modify config options using the command-line")
    parser.add_argument('--resume', type=int, default=None, help='which epoch to resume')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config(args).getcfg()
    exp_suffix = os.path.basename(args.cfg).split('.')[0]
    logger = Logger(os.path.join(cfg.LOG_DIR, '.'.join([cfg.SNAPSHOT_PREF, cfg.MODEL.NAME, exp_suffix, cfg.TRAIN.OPTIMIZER, str(cfg.TRAIN.LR)])))

    logger.log(pprint.pformat(args))
    logger.log(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(','.join([str(x) for x in cfg.GPUS]))
 
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    # torch.backends.cudnn.flags(enabled=False)

    # define data and model
    train_loader = customer_data_loader(cfg, cfg.DATASET.TRAIN_SET)
    val_loader = customer_data_loader(cfg, cfg.DATASET.TEST_SET)
    net = customer_net_loader(cfg=cfg)
    model_trainer = Trainer(net, cfg=cfg, logger=logger, bn_loader=train_loader)

    if cfg.IS_TRAIN:
        model_trainer.train(train_loader, val_loader=val_loader, which_epoch=args.resume)
    else:
        model_trainer.test(val_loader, weight_file=cfg.TEST.LOAD_WEIGHT)


if __name__=='__main__':
    main()
