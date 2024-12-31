import os
import gc
import time
import math
import glob
import shutil
import logging
import inspect
import operator
import datetime
import numpy as np
from copy import deepcopy
from collections import namedtuple
from collections import OrderedDict

import torch
from torch import nn
import torch.optim as optim
from torch.optim.sgd import SGD

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)
    

def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct fanout calculation w/ group convs

    FIXME change fix_group_fanout to default to True if experiments show better training results

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        if m.bias is not None:
            m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)


def drop_path(inputs, training=False, drop_path_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_path_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def save_checkpoint(checkpoint_file, net, epoch, optim, gs, is_parallel=True):
    checkpoint_dict = {
        'epoch': epoch,
        'global_step': gs,
        'optimizer': optim.state_dict(),
        'state_dict': net.module.state_dict() if is_parallel else net.state_dict()
    }
    torch.save(checkpoint_dict, checkpoint_file, _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_file, is_parallel=False):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    if is_parallel:
        w_dict = checkpoint['state_dict']
        w_dict = {'module.' + k: v for k, v in w_dict.items()}
    else:
        w_dict = checkpoint['state_dict']
    return w_dict, checkpoint


def get_optimizer(cfg, model, policies=None):
    if policies is None:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER=='adamw':
            optimizer = optim.AdamW(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR
            )
        else:
            raise(KeyError, '%s not supported yet...'%cfg.TRAIN.OPTIMIZER)
    else:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                policies,
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                policies,
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'adamw':
            optimizer = AdamW(
                policies,
                lr=cfg.TRAIN.LR
            )
        else:
            raise(KeyError, '%s not supported yet...'%cfg.TRAIN.OPTIMIZER)

    return optimizer


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def load_org_weights(model, pretrained, repeat_bn=True):
    pre_dict = torch.load(pretrained, map_location='cpu')
    tic = time.time()
    model_dict = model.state_dict()
    if 'state_dict' in pre_dict.keys():
        pre_dict = pre_dict['state_dict']
    for name in model_dict.keys():
        if 'num_batches_tracked' in name:
            continue
        is_null = True
        # TODO: Load BatchNorm Layers
        if repeat_bn and 'aux_bn' in name:
            tmp_name = name.split('aux_bn')[0]+name.split('.')[-1]
        else:
            tmp_name = name

        try:
            if model_dict[name].shape == pre_dict[tmp_name].shape:
                model_dict[name] = pre_dict[tmp_name]
                is_null = False
            else:
                print('size mismatch for %s, expect (%s), but got (%s).'
                        % (name, ','.join([str(x) for x in model_dict[name].shape]),
                            ','.join([str(x) for x in pre_dict[tmp_name].shape])))
            continue
        except KeyError:
            pass
        if is_null:
            print('Do not load %s' % name)

    model.load_state_dict(model_dict)
    print('Load pre-trained weightin %.4f sec.' % (time.time()-tic))


class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume='', log_info=True, resume_strict=True):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        self.log_info = log_info
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self.strict = resume_strict
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict, strict=self.strict)
            if self.log_info:
                logging.info("Loaded state_dict_ema")
        else:
            logging.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            if not parameter.requires_grad:
                continue
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))

# def get_video_container(client, path_to_vid, multi_thread_decode=False, backend="pyav"):
#     """
#     Given the path to the video, return the pyav video container.
#     Args:
#         path_to_vid (str): path to the video.
#         multi_thread_decode (bool): if True, perform multi-thread decoding.
#         backend (str): decoder backend, options include `pyav` and
#             `torchvision`, default is `pyav`.
#     Returns:
#         container (container): video container.
#     """
#     if backend == "torchvision":
#         if client:
#             video_bytes = client.get(path_to_vid)
#             container = memoryview(video_bytes)
#         else:
#             with open(path_to_vid, "rb") as fp:
#                 container = fp.read()
#         return container
#     elif backend == "pyav":
#         container = av.open(path_to_vid)
#         if multi_thread_decode:
#             # Enable multiple threads for decoding.
#             container.streams.video[0].thread_type = "AUTO"
#         return container
#     else:
#         raise NotImplementedError("Unknown backend {}".format(backend))