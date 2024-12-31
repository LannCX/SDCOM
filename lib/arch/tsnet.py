import random
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEBUG = True
if _DEBUG:
    from posixpath import pardir
    import os, sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,pardir)

from arch.ops import DyChLinear, apply_differentiable, to_var
from arch.policy_net import PolicyNet
from arch.dy_resnet import *
from arch.dy_tsmresnet import dytsmresnet50
from utils.helpers import set_exist_attr, load_org_weights


class TSNet(nn.Module):
    def __init__(self, net='resnet50', num_classes=200, 
        num_segment=8, dropout=0.2, 
        pre_trained='', feat_pretrained='', policy_pretrained='', p_net='mobilenetv2',
        channel_list=[0, 0.25, 0.5, 0.75, 1.0], random_init=True):
        super(TSNet, self).__init__()
    # def __init__(self, cfg, random_init=False):
    #     super(TSNet, self).__init__()
        # net = cfg.MODEL.FEAT_NET
        # p_net = cfg.MODEL.PRIM_NET
        # dropout = cfg.MODEL.DROPOUT
        # pre_trained = cfg.MODEL.PRETRAINED
        # feat_pretrained = cfg.MODEL.FEAT_PRETRAINED
        # policy_pretrained = cfg.MODEL.POLICY_PRETRAINED
        # num_classes = cfg.MODEL.NUM_CLASSES
        # num_segment = cfg.DATASET.N_SEGMENT
        # channel_list = list(cfg.MODEL.CHANNEL_LIST)

        # self.cfg = cfg
        self.num_segment = num_segment  # number of frames
        self.channel_ratio = -1
        self.channel_choice = -1
        self.channel_list = channel_list
        self.ch_score = None
        self.num_classes = num_classes

        if net=='resnet18':
            self.backbone = resnet18()
            self.last_dim = 2048
        elif net=='resnet50':
            self.backbone = resnet50()
            self.last_dim = 2048
        elif net=='tsm-resnet50':
            self.backbone = dytsmresnet50(n_segment=num_segment)
            self.last_dim = 2048
        else:
            raise(KeyError, '%s is not supported yet.' % net)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.policy_net = PolicyNet(len(self.channel_list), num_classes, num_segment=self.num_segment, net_name=p_net, dropout=dropout)
        self.classifier = DyChLinear(self.last_dim, num_classes)
        self.cat_dim = self.last_dim+self.policy_net.feat_dim
        self.final_classifier = PoolingClassifier(
            input_dim=self.cat_dim,
            num_segments=self.num_segment,
            num_classes=num_classes,
            dropout=dropout
        )

        # /********** inference/training mode ******************/
        # 'supernet': supernet training
        # 'policy': policy network training
        # 'inference': dynamic inference
        self.set_stage('static_inference')
        # self.set_stage(self.cfg.TRAIN.STAGE)
        self.set_module_channel_list()
        
        if not random_init:
            if pre_trained:
                load_org_weights(self, pre_trained, repeat_bn=False)
                if self.stage=='policy':
                    load_org_weights(self.policy_net.feat_net, policy_pretrained)
            else:
                if feat_pretrained:
                    load_org_weights(self.backbone, feat_pretrained) 
                if policy_pretrained:
                    load_org_weights(self.policy_net.feat_net, policy_pretrained)
            
        # if self.cfg.TRAIN.STAGE=='supernet':
        #     self.freeze_policynet()
        # else:
        #     self.freeze_backbone()

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        identity = x

        if self.stage=='policy':
            policy_feat, policy_logit, g_soft, g_hard, g_idx = self.policy_net(x)
            self.set_module_channel_choice(g_hard)
            x = self.backbone.feature_forward(x)
            x = apply_differentiable(x, g_soft,self.channel_list, self.last_dim)
            x = self.avgpool(x)
            slim_feat = x.view(x.size(0), -1)
            slim_logit = self.classifier(slim_feat)
            # slim_logit = apply_differentiable(slim_logit, g_soft,self.channel_list, self.num_classes, logit=True)

            cat_feat = torch.cat([policy_feat, slim_feat], dim=-1)
            cat_feat_v = cat_feat.view(-1, self.num_segment, self.cat_dim)
            cat_logit, cat_pred = self.final_classifier(cat_feat_v)
            return cat_logit, cat_pred, policy_logit, slim_logit, g_hard, cat_feat
        elif self.stage=='supernet':
            assert self.channel_ratio >= 0, 'Please set valid channel ratio first.'
            x = self.backbone.feature_forward(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            out_logit = self.classifier(x)
            out_logit = out_logit.view(b,self.num_segment,-1).mean(dim=1)
            return out_logit
        elif self.stage=='inference':
            policy_feat, policy_logit, g_soft, g_hard, g_idx = self.policy_net(x)
            feat_list = []
            for idx, ch_idx in enumerate(list(g_idx)):
                r_ch = self.channel_list[ch_idx]
                img = identity[idx].unsqueeze(0)
                self.set_channel_ratio(r_ch)
                x = self.backbone.feature_forward(img)
                x = self.avgpool(x)
                slim_feat = x.view(x.size(0), -1)
                pad_feat = torch.cat([slim_feat, torch.zeros(1,self.last_dim-slim_feat.size(-1)).cuda()], dim=-1)
                feat_list.append(pad_feat)
            img_feat = torch.cat(feat_list, dim=0)
            cat_feat = torch.cat([policy_feat, img_feat], dim=-1)
            cat_feat = cat_feat.view(-1, self.num_segment, self.cat_dim)
            cat_logit, cat_pred = self.final_classifier(cat_feat)
            return cat_logit, cat_pred, g_hard, g_idx, policy_feat, img_feat
        elif self.stage=='static_inference':
            policy_feat, policy_logit, g_soft, g_hard, g_idx = self.policy_net(x)
            self.set_channel_ratio(0.25)
            x = self.backbone.feature_forward(x)
            x = self.avgpool(x)
            slim_feat = x.view(x.size(0), -1)
            pad_feat = torch.cat([slim_feat, torch.zeros(b*t,self.last_dim-slim_feat.size(-1)).cuda()], dim=-1)
            # feat_list.append(pad_feat)
            # img_feat = torch.cat(feat_list, dim=0)
            cat_feat = torch.cat([policy_feat, pad_feat], dim=-1)
            cat_feat = cat_feat.view(-1, self.num_segment, self.cat_dim)
            cat_logit, cat_pred = self.final_classifier(cat_feat)
            return cat_logit, cat_pred, g_hard, g_idx
        else:
            raise(KeyError, 'Not supported stage %s.' % self.stage)

    def set_stage(self, stage):
        self.stage=stage
        for m in self.modules():
            set_exist_attr(m, 'stage', stage)
    
    def set_module_channel_list(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_list', self.channel_list)
    
    def set_module_channel_choice(self, channel_choice):
        self.channel_choice = channel_choice
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', channel_choice)

    def set_channel_ratio(self, channel_ratio):
        # set channel ratio manually
        self.channel_ratio = channel_ratio
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_ratio', channel_ratio)

    def freeze_backbone(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.apply(fix_bn)
        for name, param in self.classifier.named_parameters():
            param.requires_grad = False

    def freeze_policynet(self):
        for name, param in self.policy_net.named_parameters():
            param.requires_grad = False
        for name, param in self.final_classifier.named_parameters():
            param.requires_grad = False

    def set_ch_score(self, score):
        setattr(self, 'ch_score', score)
    
    def get_optim_policies(self):
        return [{'params': self.policy_net.gate.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': self.cfg.TRAIN.GATE_LR, 'decay_mult': 1,
                 'name': "policy_gate"}] \
               + [{'params': self.policy_net.feat_net.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': self.cfg.TRAIN.POLICY_LR, 'decay_mult': 1,
                   'name': "policy_cnn"}] \
               + [{'params': self.backbone.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': 0.1, 'decay_mult': 1, 'name': "backbone_layers"}] \
               + [{'params': self.classifier.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': 1, 'decay_mult': 1, 'name': "backbone_fc"}] \
               + [{'params': self.final_classifier.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': self.cfg.TRAIN.CLS_FC_LR, 'decay_mult': 1,
                   'name': "pooling_classifier"}]


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), dim=1)
        return x.max(dim=1)[0]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]
        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PoolingClassifier(nn.Module):
    def __init__(self, input_dim, num_segments, num_classes, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_pooling = MaxPooling()
        self.mlp = MultiLayerPerceptron(input_dim)
        self.num_segments = num_segments
        self.classifiers = nn.ModuleList()
        for m in range(self.num_segments):
            self.classifiers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(4096, self.num_classes)
            ))

    def forward(self, x):
        _b = x.size(0)
        x = x.view(-1, self.input_dim)
        z = self.mlp(x).view(_b, self.num_segments, -1)
        logits = torch.zeros(_b, self.num_segments, self.num_classes).cuda()
        cur_z = z[:, 0]
        for frame_idx in range(0, self.num_segments):
            if frame_idx > 0:
                cur_z = self.max_pooling(z[:, frame_idx], cur_z)
            logits[:, frame_idx] = self.classifiers[frame_idx](cur_z)
        last_out = logits[:, -1, :].reshape(_b, -1)
        logits = logits.view(_b * self.num_segments, -1)
        return logits, last_out


if __name__ =='__main__':
    import time
    from utils.slim_profiling import add_flops, add_mac_hooks
    from utils.thop.profile import clever_format
    d_in = torch.zeros(1,16,3,224,224).cuda()
    model = TSNet(num_segment=8).cuda()
    # model = resnet50()
    # model.apply(lambda m: add_mac_hooks(m))

    for m in model.modules():
        set_exist_attr(m, 'stage', 'static_inference')
    # for n, m in model.named_modules():
    #     set_exist_attr(m, 'channel_ratio', 0.5)
    
    model.eval()
    tic = time.time()
    for _ in range(100):
        out = model(d_in)
    avg_time = (time.time()-tic)/100
    print(8/avg_time)

    # running_flops = 16*add_flops(model)
    # print(clever_format([running_flops]))
    # print(clever_format([running_flops]))
    