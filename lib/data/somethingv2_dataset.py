import os
import sys
import json
import pickle
import pandas as pd
from tqdm import tqdm
sys.path.append('../')
from data.base_dataset import BaseDataset
import pdb

class Somethingv2Dataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(Somethingv2Dataset, self).__init__(cfg, split)

        data_root = cfg.DATASET.ROOT
        anno_file = cfg.DATASET.ANNO_FILE
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'something-something-'+self.version
        label_dict = json.load(open(class_file % self.name))
        self.class_2_id = {k: int(v) for k, v in label_dict.items()}
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}
        self.target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
        
        print('Loading annotations...')
        save_path = '%s-%s_data.pkl' % (self.name, split)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.anno = pickle.load(f)
        else:
            #pdb.set_trace()
            sample_list = json.load(open(anno_file % (self.name, split),'r',encoding='utf-8'))
            #sample_list = json.load(open(anno_file % (self.name, split)))
            nbar = tqdm(total=len(sample_list))
            for items in sample_list:
                nbar.update(1)
                vid_id = items['id']
                vid_path = os.path.join(data_root, vid_id+'.pkl')
                template = items['template']
                label = template.replace('[','').replace(']','')
                self.anno.append({'id': vid_id, 'path': vid_path, 'label': label})
                # self.anno.append({'id': vid_id, 'path': vid_path, 'label': label, 'nf': len(os.listdir(vid_path))})
            nbar.close()
            #with open(save_path, 'wb') as f:
            #    pickle.dump(self.anno, f)

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
