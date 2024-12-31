from posixpath import pardir
import re
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import normal_, constant_
from torchvision.models import resnet101
import pdb

_DEBUG = True
if _DEBUG:
    from posixpath import pardir
    import os, sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,pardir)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_segment=8):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, pretrained, num_classes, n_segment=16):
        super(ResNet, self).__init__()
        net_type = 'resnet50'

        if net_type=='resnet50':
            layers = [3,4,6,3]
        elif net_type=='resnet101':
            layers = [3,4,23,3]
        else:
            raise(KeyError, 'not supported net type: %s' % net_type)

        self.inplanes = 64
        block = Bottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], n_segment=n_segment)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, n_segment=n_segment)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, n_segment=n_segment)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, n_segment=n_segment)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.gru = nn.GRU(input_size=512*block.expansion, hidden_size=1024, bias=True, batch_first=True)
        #self.fc = nn.Linear(1024, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(1)

        if pretrained:
            self.load_org_weights(torch.load(pretrained, map_location='cpu'))
            # self.load_org_weights(model_zoo.load_url(model_urls['resnet50']))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    std = 0.001
                    normal_(m.weight, 0, std)
                    constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, n_segment=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_segment=n_segment))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_segment=n_segment))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape((-1,)+x.shape[-3:])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean([2,3])
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = x.reshape((-1, self.n_segment) + x.shape[-1:]).mean(dim=1)
        return x

    def load_org_weights(self, pre_dict):
        tic = time.time()
        model_dict = self.state_dict()
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name in model_dict.keys():
            if 'num_batches_tracked' in name:
                continue
            tmp_name = name if name in pre_dict.keys() else 'extractor.'+name 
            is_null = True
            #pdb.set_trace()
            try:
                if model_dict[name].shape == pre_dict[tmp_name].shape:
                    model_dict[name] = pre_dict[tmp_name]
                    is_null = False
                else:
                    print('size mismatch for %s, expect (%s), but got (%s).'
                          % (name, ','.join([str(x) for x in model_dict[name].shape]),
                             ','.join([str(x) for x in pre_dict[name].shape])))
                continue
            except KeyError:
                pass
            if is_null:
                print('Do not load %s' % name)

        self.load_state_dict(model_dict)

        print('Load pre-trained weightin %.4f sec.' % (time.time()-tic))


def runtime_test():
    import os
    from tqdm import tqdm
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3' #str(','.join([str(x) for x in cfg.GPUS]))
    model = ResNet(pretrained='',num_classes=200, n_segment=16)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    test_n = 100
    nbar = tqdm(total=test_n)
    tic = time.time()
    for _ in range(test_n):
        nbar.update(1)
        dump_data = torch.randn(16*16,3,224,224).cuda()
        output = model(dump_data)
    nbar.close()
    toc = time.time()
    total_time = toc-tic
    print(total_time)
    print('%.2f video/s.'% (test_n*16/total_time))

if __name__=='__main__':
    from utils.thop import profile, clever_format

    # d_in = torch.randn(16,3,224,224)
    # m = ResNet(pretrained='',num_classes=200, n_segment=16)
    # macs, params = profile(m, inputs=(d_in,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('Macs:' + macs + ', Params:' + params)
    runtime_test()