import os
import pickle
import matplotlib.pyplot as plt
from numpy.random import randint
from torch.utils.data import Dataset

from decord import VideoReader
from data.video_transforms import *
import torchvision.transforms as transf
import pdb
import random
import numpy as np
import torch
import torchvision.io as io


class BaseDataset(Dataset):
    def __init__(self, cfg, split):
        self.split = split if split == 'train' else 'test'
        self.cfg = cfg
        self.eval = not cfg.IS_TRAIN
        self.test_crops = cfg.TEST.NUM_CROPS
        self.n_segment = cfg.DATASET.N_SEGMENT
        self.version = cfg.DATASET.VERSION
        self.is_flip = cfg.DATASET.IS_FLIP
        self.scale_size = cfg.DATASET.SCALE_SIZE
        self.crop_size = cfg.DATASET.CROP_SIZE
        self.data_type = cfg.DATASET.DATA_TYPE
        self.image_tmpl = cfg.DATASET.IMG_FORMAT
        self.samp_mode = cfg.DATASET.SAMP_MODE  # dense, global
        self.resample = cfg.DATASET.RESAMPLE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        
        if self.resample:
            self.new_order = cal_seq_order(self.n_segment)
        if 'TDN'==cfg.MODEL.FEAT_NET:
            self.new_length = 5
        else:
            self.new_length = 1

        self.anno = []
        self.id_2_class = {}
        self.class_2_id = {}
        self.t_step = 8
        self._num_retries = 3

        # Transformation
        self.roll = True if 'Inception' in cfg.MODEL.NAME else False
        self.div = not self.roll
        # self.is_flip = False if 'something' in cfg.DATASET.NAME else True
        normalize = IdentityTransform() if 'InceptionV1' in cfg.MODEL.NAME else GroupNormalize(cfg.DATASET.MEAN, cfg.DATASET.STD)
        if 'somethingv2'==cfg.DATASET.NAME:
            target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
            flipping = GroupRandomHorizontalFlip_sth(target_transforms) if self.is_flip and self.split=='train' else IdentityTransform()
        else:
            flipping = GroupRandomHorizontalFlip() if self.is_flip and self.split=='train' else IdentityTransform()

        if self.eval:
            if self.test_crops==1:
                scale_crop = transf.Compose([GroupScale(self.scale_size), GroupCenterCrop(self.crop_size)])
            elif self.test_crops==3:
                scale_crop = GroupFCNSample_0(256)
            else:
                raise (KeyError, 'Not supported inference mode {}'.format(self.samp_mode))
        else:
            scale_crop = GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66], fix_crop=True, more_fix_crop=True) \
                if self.split == 'train' else transf.Compose([GroupScale(self.scale_size), GroupCenterCrop(self.crop_size)])
        
        self.transform = transf.Compose([
            scale_crop,
            flipping,
            Stack(self.roll),
            ToTorchFormatTensor(div=self.div),
            normalize
        ])

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')

    def _sample_indices(self, num_frames, need_f):
        if self.samp_mode=='dense':
            if num_frames <= need_f:
                indices = list(range(1, num_frames + 1))
                for i in range(need_f - num_frames):
                    indices.append(num_frames)
            else:
                start = random.randint(1, num_frames - need_f + 1)
                indices = list(range(start, start + need_f))
            return indices
        elif self.samp_mode=='global':
            average_duration = (num_frames-self.new_length+1) // need_f
            if average_duration > 0:
                offsets = np.multiply(list(range(need_f)), average_duration) + randint(average_duration, size=need_f)
            elif num_frames > need_f:
                offsets = np.sort(randint(num_frames-self.new_length+1, size=need_f))
            else:
                offsets = np.zeros((need_f,))
                #offsets = np.array(list(range(num_frames-self.new_length+1)) + [num_frames - self.new_length] * (need_f-num_frames+self.new_length-1))
            return offsets + 1
        else:
            raise(KeyError, 'Not supported sampling mode %s'%self.samp_mode)

    def _get_val_indices(self, num_frames, need_f):
        if self.samp_mode=='dense':
            if num_frames <= need_f:
                indices = list(range(1, num_frames + 1))
                for i in range(need_f - num_frames):
                    indices.append(num_frames)
            else:
                start = (num_frames - need_f) // 2
                indices = list(range(start, start + need_f))
            return indices
        elif self.samp_mode=='global':
            if num_frames > need_f + self.new_length - 1:
                tick = (num_frames-self.new_length+1) / float(need_f)
                # offsets = np.array([int(tick * x) for x in range(need_f)])
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(need_f)])
            else:
                offsets = np.zeros((need_f,))
                #offsets = np.array(list(range(num_frames-self.new_length+1)) + [num_frames-self.new_length] * (need_f-num_frames+self.new_length-1))
            return offsets + 1
        else:
            raise(KeyError, 'Not supported sampling mode %s'%self.samp_mode)

    def _get_test_indices(self, num_frames, need_f):
        if self.samp_mode=='dense':
            indices = []
            if num_frames <= need_f:
                indices = list(range(1, num_frames + 1))
                for i in range(need_f - num_frames):
                    indices.append(num_frames)
            else:
                for start in range(1, num_frames - need_f + 1, self.t_step):
                    indices.extend(list(range(start, start + need_f)))
            return indices
        elif self.samp_mode=='global':
            if num_frames > need_f + self.new_length - 1:
                tick = (num_frames-self.new_length+1) / float(need_f)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(need_f)])
            else:
                offsets = np.zeros((need_f,))
                #offsets = np.array(list(range(num_frames-self.new_length+1)) + [num_frames-self.new_length] * (need_f-num_frames+self.new_length-1))
            return offsets + 1
        else:
            raise(KeyError, 'Not supported sampling mode %s'%self.samp_mode)

    def get_indices(self, nf, need_f):
        indices = self._sample_indices(nf, need_f) if self.split=='train' else \
            self._get_test_indices(nf, need_f) if self.eval else \
                self._get_val_indices(nf, need_f)
        return indices

    def __getitem__(self, index):
        for i_try in range(self._num_retries):
            vid_info = self.anno[index]
            if self.data_type=='video':
                vr = None
                try:
                    vr = VideoReader(vid_info['path'])
                except Exception as e:
                    print('Failed to load {} with error {}'.format(vid_info['path'], e))
                if vr is None:
                    index = random.randint(0, len(self.anno)-1)
                    continue
                nf = len(vr)
                indices = self.ext_indices(self.get_indices(nf, self.n_segment), nf)          
                indices = [x-1 for x in indices]
                img_group = vr.get_batch(indices).asnumpy()
                img_group = [Image.fromarray(img) for img in img_group]
                # vr.close()
            elif self.data_type=='img':
                nf = vid_info['nf']
                indices = self.get_indices(nf, self.n_segment)
                indices = self.ext_indices(indices, nf)
                img_group = [self._load_image(vid_info['path'], int(ind)) for ind in indices]
            elif self.data_type=='pkl':
                video_pkl = pickle.load(open(vid_info['path'],'rb'))
                nf = len(video_pkl)
                indices = [x-1 for x in self.get_indices(nf, self.n_segment)]
                if len(indices)!=self.n_segment:
                    raise KeyError(vid_info['path']+' frame number:%d' % nf)
                indices = self.ext_indices(indices, nf)
                img_group = [Image.open(video_pkl[ind]).convert('RGB') for ind in indices]
            else:
                raise KeyError('Not supported data type: {}.'.format(self.data_type))
            # img_tensor = self.transform(img_group)  # [T,C,H,W]
            if self.name in ['anet', 'diving48', 'fcvid']:
                img_tensor, label = self.transform((img_group, vid_info['label']))  # [T,C,H,W]
            else:
                img_tensor, label = self.transform((img_group, self.class_2_id[vid_info['label']]))  # [T,C,H,W]        
            if self.resample:
                inds = torch.tensor(self.new_order, dtype=torch.long)
                img_tensor = torch.index_select(img_tensor, 0, inds)

            if 'InceptionV1' in self.cfg.MODEL.NAME:
                img_tensor = img_tensor/255. * 2 - 1.  # Inception normalization
            
            return img_tensor, label, vid_info['id']
        else:
            raise RuntimeError(
                "Failed to fetch video idx {} from {}; after {} trials.".format(
                index, vid_info['path'], self._num_retries
                )
            )

    def __len__(self):
        return len(self.anno)
    
    def ext_indices(self, indices, num_frames):
        new_indices = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                new_indices.append(p)
                if p+1 < num_frames:
                        p += 1
        return new_indices

def cal_seq_order(num_frames):
    S = [np.floor((num_frames-1)/2), 0, num_frames-1]
    q = 2
    while len(S)<num_frames:
        interval = np.floor(np.linspace(0,num_frames-1, q+1))
        for i in range(0, len(interval)-1):
            a = interval[i]
            b = interval[i+1]
            ind = np.floor((a+b)/2)
            if not ind in S:
                S.append(ind)
        q *=2
    S = [int(s) for s in S]
    return S
    
