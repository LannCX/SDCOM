import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """
    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(
            input_channels, input_channels,
            kernel_size=3, padding=1, groups=input_channels,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = False
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


class DyShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8, n_div=8):
        super(DyShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        # self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(
            input_channels, input_channels,
            kernel_size=3, padding=1, groups=input_channels,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        self.conv.weight.requires_grad = False
        self.conv.weight.data.zero_()
        self.stage = 'supernet'
        self.channel_ratio = 1.
        self.channel_choice = -1
        self.channel_list = [0., 0.25, 0.5, 0.75, 1.]

    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        
        # if self.stage=='policy':
        #     fold = None
        #     self.conv.weight.data[:fold, 0, 2] = 1 # shift left
        #     self.conv.weight.data[fold: 2 * fold, 0, 0] = 1 # shift right
        #     if 2*fold < self.input_channels:
        #         self.conv.weight.data[2 * fold:, 0, 1] = 1 # fixed
            
        #     x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        # else:
        fold = c // self.fold_div
        self.conv.weight.data[:fold, 0, 2] = 1 # shift left
        self.conv.weight.data[fold: 2*fold, 0, 0] = 1 # shift right
        if 2*fold < self.input_channels:
            self.conv.weight.data[2*fold:, 0, 1] = 1 # fixed

        weight = self.conv.weight[:c,:c]
        output = F.conv1d(x, weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, c)
        
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


class MLP(nn.Module):
    def __init__(self, in_planes, hidden_size=100):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(in_planes, hidden_size)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(hidden_size, in_planes)
        
    def forward(self, x):
        x = self.relu(self.first_layer(x))
        x = self.out_layer(x)
        return torch.sigmoid(x)


class DyChConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
        in_ch_static=False, out_ch_static=False,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding=0,
        padding_mode='zeros'):
        super(DyChConv2d, self).__init__(in_channels, out_channels, kernel_size, 
            stride=stride, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding=padding,
            padding_mode=padding_mode)
        self.channel_ratio = 1.
        self.in_ch_static = in_ch_static
        self.out_ch_static = out_ch_static

        # For calculating FLOPs 
        self.running_inc = self.in_channels if self.in_ch_static else None
        self.running_outc = self.out_channels if self.out_ch_static else None
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = groups

        self.stage = 'supernet'
        self.channel_choice = -1
        self.channel_list = [0., 0.25, 0.5, 0.75, 1.]
    
    def forward(self, x):
        ch_lst_tensor = torch.from_numpy(np.array(self.channel_list)).unsqueeze(-1).float().to(x.device) # K,1
        if self.stage=='policy':
            assert isinstance(self.channel_choice, torch.Tensor), 'Please set a valid channel_choice first.'
            if not self.in_ch_static:
                num_ch = self.in_channels*torch.matmul(self.channel_choice, ch_lst_tensor)
                num_ch = torch.where(num_ch==0, torch.ones_like(num_ch), num_ch)
                self.running_inc = num_ch.mean().item()
            if not self.out_ch_static:
                num_ch = self.out_channels*torch.matmul(self.channel_choice, ch_lst_tensor)
                num_ch = torch.where(num_ch==0, torch.ones_like(num_ch), num_ch)
                self.running_outc = num_ch.mean().item()
            
            # Training with channel mask
            weight = self.weight
            output = F.conv2d(x,
                              weight,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
            
            if not self.out_ch_static:
                output = apply_differentiable(output, self.channel_choice.detach(), \
                                                self.channel_list, self.out_channels)
        else:
            # For training supernet and deterministic dynamic inference
            if not self.in_ch_static:
                self.running_inc = x.size(1)
                assert self.running_inc == max(1, int(self.in_channels*self.channel_ratio)), \
                'running input channel %d does not match %dx%.2f'%(self.running_inc, self.in_channels, self.channel_ratio)
            
            # since 0 channel is invalid for calculating output tensors, we use 1 channel to approximate skip operation
            if not self.out_ch_static:
                self.running_outc = max(1, int(self.out_channels*self.channel_ratio))
            weight = self.weight[:self.running_outc,:self.running_inc]
            bias = self.bias[:self.running_outc] if self.bias is not None else None
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output


class DyChBatchNorm2d(nn.Module):
    def __init__(self, in_planes, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(DyChBatchNorm2d, self).__init__()
        self.in_planes = in_planes
        self.channel_choice = -1
        self.channel_list = [0, 0.25, 0.5, 0.75, 1.]
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(max(1, int(in_planes*ch)), affine=False) for ch in self.channel_list[:-1]
        ])
        self.aux_bn.append(nn.BatchNorm2d(int(self.channel_list[-1]*in_planes),
                                          eps=eps,
                                          momentum=momentum,
                                          affine=affine))
        self.affine = affine
        self.stage = 'supernet'
    
    def set_zero_weight(self):
        if self.affine:
            nn.init.zeros_(self.aux_bn[-1].weight)
    
    @property
    def weight(self):
        return self.aux_bn[-1].weight

    @property
    def bias(self):
        return self.aux_bn[-1].bias

    def forward(self, x):
        running_inc = x.size(1)
        idx_offset = 5-len(self.channel_list)
        if self.stage=='policy':
            assert isinstance(self.channel_choice, torch.Tensor), 'Please set a valid channel_choice first.'
            
            # Training with channel mask
            running_mean = torch.zeros_like(self.aux_bn[-1].running_mean).repeat(len(self.channel_list), 1)
            running_var = torch.zeros_like(self.aux_bn[-1].running_var).repeat(len(self.channel_list), 1)
            for i in range(len(self.channel_list)):
                n_ch = max(1, int(self.in_planes*self.channel_list[i]))
                running_mean[i, :n_ch] += self.aux_bn[i+idx_offset].running_mean
                running_var[i, :n_ch] += self.aux_bn[i+idx_offset].running_var
            running_mean = torch.matmul(self.channel_choice.detach(), running_mean)[..., None, None].expand_as(x)
            running_var = torch.matmul(self.channel_choice.detach(), running_var)[..., None, None].expand_as(x)
            weight = self.weight[:running_inc] if self.affine else None
            bias = self.bias[:running_inc] if self.affine else None
            x = (x - running_mean) / torch.sqrt(running_var + self.aux_bn[-1].eps)
            x = x * weight[..., None, None].expand_as(x) + bias[..., None, None].expand_as(x)
            return apply_differentiable(x, self.channel_choice.detach(), self.channel_list, self.in_planes)
        else:
            running_channel_ratio = 0. if running_inc==1 else running_inc/self.in_planes
            assert running_channel_ratio in self.channel_list, 'Current channel ratio %f is not existed!'%running_channel_ratio
            idx = self.channel_list.index(running_channel_ratio)
            running_mean = self.aux_bn[idx].running_mean
            running_var = self.aux_bn[idx].running_var
            weight = self.aux_bn[-1].weight[:running_inc] if self.affine else None
            bias = self.aux_bn[-1].bias[:running_inc] if self.affine else None
            return F.batch_norm(x,
                                running_mean,
                                running_var,
                                weight,
                                bias,
                                self.training,
                                self.aux_bn[-1].momentum,
                                self.aux_bn[-1].eps)


class DyChLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super(DyChLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)
        self.stage = 'supernet'

    def forward(self, x):
        self.running_inc = x.size(1)
        self.running_outc = self.out_features
        weight = self.weight[:,:self.running_inc]
        
        return F.linear(x, weight, self.bias)


def apply_differentiable(x, channel_choice, channel_list, in_channels, logit=False):
    ret = torch.zeros_like(x)
    for idx in range(len(channel_list)):
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        n_ch = max(1, int(in_channels*channel_list[idx]))
        if logit:
            ret[:, :n_ch] += x[:, :n_ch] * (
                channel_choice[:, idx, None].expand_as(x[:, :n_ch]))
        else:
            ret[:, :n_ch] += x[:, :n_ch] * (
                channel_choice[:, idx, None, None, None].expand_as(x[:, :n_ch]))
    return ret


def gumbel_softmax(logits, tau=1, dim=1):
    """ See `torch.nn.functional.gumbel_softmax()` """

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return y_soft, ret, index


if __name__=='__main__':
    dummy_in = torch.randn(4,64,12,12)
    ch_lst = [0,0.25,0.5,0.75,1.0]
    ch_choice = torch.from_numpy(np.array([[0,1,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,1,0,0]]))
    apply_differentiable(dummy_in, ch_choice, ch_lst, 64)
    print('OK')
