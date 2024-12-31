import random
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEBUG = False
if _DEBUG:
    from posixpath import pardir
    import os, sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,pardir)

from arch.mobilenet import mobilenet_v2
from torchvision.models import resnet50
from arch.ops import gumbel_softmax


class PolicyNet(nn.Module):
    def __init__(self, num_ch, num_classes, num_segment, temperature=5., dropout=0.8, net_name='mobilenetv2', pre_trained=''):
        super(PolicyNet, self).__init__()
        self.temperature = temperature
        self.net_name = net_name
        if self.net_name=='mobilenetv2':
            self.feat_net = mobilenet_v2()
            self.feat_dim = self.feat_net.last_channel
            spatial_size = 49 #7*7
        elif self.net_name=='resnet50':
            self.excute_layers = [
                'conv1',
                'bn1',
                'relu',
                'maxpool',
                'layer1',
                'layer2',
                'layer3',
                'layer4',
                ]
            self.feat_net = resnet50()
            self.feat_dim = self.feat_net.fc.in_features
            spatial_size = 9 #3*3
        else:
            raise RuntimeError('Net %s is not supported!'%self.net_name)

        self.feat_net.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        self.gate = Gate(self.feat_dim, num_ch, num_segment=num_segment, spatial_size=spatial_size)

    def forward(self, x):
        if self.net_name=='mobilenetv2':
            feat_map, feat = self.feat_net.get_featmap(x)
        elif self.net_name=='resnet50':
            x = F.interpolate(x,(96,96), mode='bilinear', align_corners=True)
            for end_point in self.excute_layers:
                x = getattr(self.feat_net, end_point)(x)
            feat_map, feat = x, x.mean([2,3])
        policy_logit = self.feat_net.classifier(feat)
        # if self.training:
        #     # g_logits = self.gate(feat_map.detach())
        #     g_logits = self.gate(feat_map)
        # else:
        g_logits = self.gate(feat_map)
        y_soft, ret, index = gumbel_softmax(g_logits, tau=self.temperature)

        return feat, policy_logit, y_soft, ret, index
    
    def set_temperature(self, tau):
        self.temperature = tau
    
    def decay_temperature(self, decay_ratio=None):
        if decay_ratio is not None:
            self.temperature *= decay_ratio


class Gate(nn.Module):
    def __init__(self, in_planes, num_ch, num_segment, hidden_dim=1024, spatial_size=49):
        super(Gate, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_segment = num_segment

        self.encoder = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(spatial_size*64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # reduced_ch = int(hidden_dim*0.25)
        # self.conv_reduce = nn.Linear(hidden_dim, reduced_ch)
        # self.actn = nn.ReLU(inplace=True)
        # self.conv_expand = nn.Linear(reduced_ch, hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_ch)
    
    def forward(self, x):
        mid_feat = self.encoder(x)
        mid_feat = mid_feat.view(-1, self.num_segment, self.hidden_dim)
        _b, _t, _ = mid_feat.shape
        # x = out.mean(dim=1)
        # x_reduce = self.actn(self.conv_reduce(x))
        # attn = self.conv_expand(x_reduce)
        # attn = F.softmax(attn, dim=1)
        # out = out*attn.view(_b,1,-1)
        hx = torch.zeros(self.gru.num_layers, _b, self.hidden_dim).cuda()
        self.gru.flatten_parameters()
        out, _ = self.gru(mid_feat, hx)

        out = self.fc(out.reshape(_b*_t, -1))
        return out


if __name__ =='__main__':
    from torchvision.models import resnet50
    from utils.thop import profile, clever_format
    # from utils.model_profiling import model_profiling
    d_in = torch.rand(16, 3, 224, 224).cuda()
    # m = resnet50(num_classes=200)
    m = PolicyNet(5, 200, 16).cuda()
    # model_profiling(m, 224,224,16,3,use_cuda=True,verbose=True)
    # model_profiling(m)

    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
