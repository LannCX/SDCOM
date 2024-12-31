import os
import sys
import json
import pickle
from tqdm import tqdm
sys.path.append('../')
from data.base_dataset import BaseDataset
import pdb
import torch

class FCVIDDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(FCVIDDataset, self).__init__(cfg, split)

        data_root = os.path.join(cfg.DATASET.ROOT, 'fcvid_%s_pil'%split)
        anno_file = cfg.DATASET.ANNO_FILE
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'fcvid'
        with open(class_file) as f:
            lines = f.readlines()
        self.id_2_class = {line.strip().split(',')[0]: line.strip().split(',')[1] for line in lines}
        self.class_2_id = {v: k for k, v in self.id_2_class.items()}

        print('Loading annotations...')
        with open(anno_file % split) as f:
            lines = f.readlines()
            nbar = tqdm(total=len(lines))
            miss = 0
            for n_l, line in enumerate(lines):
                nbar.update(1)
                items = line.strip().split(',')
                if int(items[1])<=3:
                    continue
                vid_id = items[0].replace('multiple/', '')
                is_null = True
                
                if 'multiple' in line:
                    parts = line.strip().split('/')
                    for cls in parts[1].split('_'):
                        vid_id = cls+'/'+parts[-1].split(',')[0]
                        vid_path = os.path.join(data_root, vid_id+'.pkl')
                        if os.path.isfile(vid_path):
                            is_null = False
                            break
                else:
                    vid_path = os.path.join(data_root, vid_id+'.pkl')
                    is_null = False if os.path.isfile(vid_path) else True
                if is_null:
                    miss+=1
                    continue
                # labels = torch.tensor([-1, -1, -1])
                labels = torch.zeros(self.num_classes, dtype=torch.float)
                objs = sorted(list(set([int(x) for x in items[2:]])))
                for l in objs:
                    labels[l] = 1.
                # for i,l in enumerate(objs):
                #     labels[i] = l

                # if labels[-2] > -1:
                #     if labels[-1] > -1:
                #         labels =  labels[torch.randperm(labels.shape[0])]
                #     else:
                #         if torch.rand(1) > 0.5:
                #             labels = labels[[0,1,2]]
                #         else:
                #             labels = labels[[1,0,2]]
                # assert labels[0] > -1
                # if n_l==500:
                #     break
                self.anno.append({'id': vid_id, 'path': vid_path, 'label': labels})
            nbar.close()
        print('Creating %s dataset completed, %d samples, miss: %d.' % (split, len(self.anno), miss))


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='../../exp_configs/fcvid-local.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from cfg.main_cfg import MainConfig as Config
    from data.data_loader import customer_data_loader

    args = parse_args()
    cfg = Config(args).getcfg()
    d_loader = customer_data_loader(cfg, cfg.DATASET.TEST_SET)

    nbar = tqdm(total=len(d_loader))
    for item in d_loader:
        assert item[0].shape[1]==40
        nbar.update(1)
    nbar.close()

