import os
import sys
import json
import pickle
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from data.base_dataset import BaseDataset
from decord import VideoReader

class KineticsDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(KineticsDataset, self).__init__(cfg, split)
        data_root = cfg.DATASET.ROOT
        anno_file = cfg.DATASET.ANNO_FILE
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'kinetics-'+self.version
        self.class_2_id = json.load(open(class_file % self.version))
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}
        # self.decode_all_video = True
        df = pd.read_csv(anno_file % (self.version, split))
        miss = 0

        print('Loading annotations...')
        # nbar = tqdm(total=df.shape[0])
        for i in range(df.shape[0]):
            # nbar.update(1)
            vid_id = df['youtube_id'][i]
            if vid_id=='0WB1oIfjA2o':
                continue
            label = df['label'][i] #.replace(' ', '_')
            is_cc = df['is_cc'][i]
            load_suc = True
            if self.data_type == 'video':
                start = str(int(df['time_start'][i])).zfill(6)
                end = str(int(df['time_end'][i])).zfill(6)
                vid_path = os.path.join(data_root, split, label, '_'.join([vid_id, start, end])+'.mp4')
                try:
                    if os.path.exists(vid_path):
                        vr = VideoReader(vid_path)
                        # imgs = vr.get_batch([0,1,2,3]).asnumpy()
                    else:
                        load_suc = False
                        # print('No such file %s.' % vid_path)
                except Exception as e:
                    miss+=1
                    load_suc = False
                if load_suc:
                    self.anno.append({'id': vid_id, 'path':vid_path, 'label': label, 'is_cc': is_cc})            
            elif self.data_type=='lmdb':
                self.anno.append({'id': vid_id, 'label': label, 'is_cc': is_cc})
            else:
                raise('Not supported data type: %s for loading Kinetics.' % self.data_type)
        # nbar.close()

        print('Creating %s dataset completed, %d samples, %d missed' % (split, len(self.anno), miss))


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='../../exp_configs/minik-local.yaml',
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
    d_loader = customer_data_loader(cfg, cfg.DATASET.TRAIN_SET)

    nbar = tqdm(total=len(d_loader))
    for item in d_loader:
        nbar.update(1)
    nbar.close()
