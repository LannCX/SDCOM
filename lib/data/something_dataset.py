import os
import sys
import json
import pickle
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from data.base_dataset import BaseDataset


class SomethingDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(SomethingDataset, self).__init__(cfg, split)

        data_root = cfg.DATASET.ROOT
        anno_file = cfg.DATASET.ANNO_FILE
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'something-something-'+self.version
        with open(class_file % self.name) as f:
            lines = f.readlines()
        self.class_2_id = {line.rstrip(): i for i, line in enumerate(lines)}
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}

        self.target_transforms = {2:4, 4:2, 41:30, 30:41, 52:66, 66:52}
        print('Loading annotations...')
        save_path = '%s-%s_data.pkl' % (self.name, split)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.anno = pickle.load(f)
        else:
            with open(anno_file % (self.name, split)) as f:
                lines = f.readlines()
                nbar = tqdm(total=len(lines))
                for line in lines:
                    nbar.update(1)
                    line = line.rstrip()
                    items = line.split(';')
                    vid_id = items[0]
                    vid_path = os.path.join(data_root, vid_id)
                    label = items[1]
                    self.anno.append({'id': vid_id, 'path': vid_path, 'label': label})
                    # self.anno.append({'id': vid_id, 'path': vid_path, 'label': label, 'nf': len(os.listdir(vid_path))})
                nbar.close()
                with open(save_path, 'wb') as f:
                    pickle.dump(self.anno, f)

        print('Creating %s dataset completed, %d samples.' % (split, len(self.anno)))


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='../../experiments/something-something.yaml',
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
        nbar.update(1)
    nbar.close()
