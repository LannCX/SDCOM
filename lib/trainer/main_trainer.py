import sys
import time
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from torch.nn import functional as F

from utils import AverageMeter
from utils.helpers import ModelEma, MetaSGD
from utils.eval import accuracy, cal_map
from utils.slim_profiling import add_flops, add_mac_hooks
from utils.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, CosineDistance, TripletLoss
from trainer.base_trainer import BaseTrainer

from arch.ops import DyChBatchNorm2d, MLP
from arch.tsnet import TSNet
from arch.tsnresnet import TSNResNet

import pdb
import pickle


class MainTrainer(BaseTrainer):
    def __init__(self, net, cfg, logger=None, bn_loader=None):
        super(MainTrainer, self).__init__(net, cfg, logger=logger)
        if self.cfg.DATASET.NAME=='fcvid':
            self.train_loss_fn = SoftTargetCrossEntropy().to(self.device)
            self.validate_loss_fn = SoftTargetCrossEntropy().to(self.device)
        else:
            self.train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.cfg.TRAIN.SMOOTHING).to(self.device)
            self.validate_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.distill_loss_fn = SoftTargetCrossEntropy().to(self.device)
        self.simliarity_loss_fn = CosineDistance().to(self.device)

        self.train_metrics['train_acc'] = AverageMeter()
        self.multi_label = True if self.cfg.DATASET.NAME in ['fcvid','anet'] else False
        self.use_concat = cfg.TRAIN.USE_CONCAT
        self.use_aux_loss = cfg.TRAIN.USE_AUX_LOSS
        self.reset_bn = cfg.TRAIN.RECALIBRATE_BN
        self.teacher_weight = cfg.TRAIN.TEACHER_PRETRAINED
        self.use_meta_net = cfg.TRAIN.USE_META_NET
        self.bn_loader = bn_loader
        self.train_stage = self.cfg.TRAIN.STAGE
        if hasattr(self.net, 'module'):
            self.channel_list = self.net.module.channel_list
        else:
            self.channel_list = self.net.channel_list
        self.use_ema = cfg.TRAIN.USE_EMA
        self.num_segment = cfg.DATASET.N_SEGMENT
        self.init_op()

    def init_op(self):
        if self.parallel:
            self._net = torch.nn.DataParallel(self._net)
        self._net = self._net.to(self.device)

        # Add EMA module
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        if self.use_ema:
            self.model_ema = ModelEma(
                self._net,
                decay=self.cfg.TRAIN.EMA_DECAY,
                device=self.device)

        if self.reset_bn:
            self.recalibrate_bn(self.bn_loader)
        
        if self.teacher_weight:
            self.teacher_net = TSNResNet(self.cfg, self.teacher_weight)
            self.teacher_net.eval()
            self.teacher_net = self.teacher_net.to(self.device)
        
        if self.use_meta_net:
            self.meta_num = self.cfg.TRAIN.META_NUM
            self.meta_net = MLP(2)
            # if self.parallel:
            #     self.meta_net = torch.nn.DataParallel(self.meta_net) #.to(self.device)
            self.meta_net = self.meta_net.to(self.device)
            self.meta_optimizer = torch.optim.Adam(self.meta_net.parameters(), 
                                                   lr=self.cfg.TRAIN.META_LR, 
                                                   weight_decay=self.cfg.TRAIN.META_WD)

    def feed_data_and_run_loss(self, data):
        input_tensor, labels = data[0], data[1]
        input_tensor = input_tensor.to(self.device)
        if self.multi_label and self.cfg.DATASET.NAME=='anet':
            labels = labels[:,0]
        labels = labels.to(self.device)
        _b = labels.shape[0]
        if self.cfg.DATASET.NAME=='fcvid':
            labels_e = labels.unsqueeze(1).expand(_b, self.num_segment, labels.size(-1)).reshape(_b*self.num_segment, -1)
        else:
            labels_e = labels.unsqueeze(1).expand(_b, self.num_segment).reshape(-1)
        
        if self.train_stage=='supernet':
            loss, out_logit = self.train_supernet(input_tensor, labels)
        else: # policy
            loss, out_logit = self.train_policy(input_tensor, labels_e)
        
        if not self.multi_label:
            acc = accuracy(out_logit.data, labels, topk=(1,))[0]
            self.train_metrics['train_acc'].update(acc.item(), input_tensor.size(0))
        return loss

    def train_supernet(self, input_tensor, target):
        guide_list = []

        for ch_idx in range(-1, -(len(self.channel_list)+1), -1):
            ch = self.channel_list[ch_idx]
            if hasattr(self.net, 'module'):
                self.net.module.set_channel_ratio(ch)
            else:
                self.net.set_channel_ratio(ch)
            out_logit = self.net(input_tensor)
            
            # Pseudo-Skip: 1 channel
            if ch==self.channel_list[0]:
                loss = self.train_loss_fn(out_logit, target)
            # All channels 
            if ch==self.channel_list[-1]:
                loss = self.train_loss_fn(out_logit, target)
                if self.use_ema:
                    with torch.no_grad():
                        if hasattr(self.model_ema.ema, 'module'):
                            self.model_ema.ema.module.set_channel_ratio(ch)
                        else:
                            self.model_ema.ema.set_channel_ratio(ch)
                        output_largest = self.model_ema.ema(input_tensor)
                    guide_list.append(output_largest)
                loss_largest = loss.mean()
                out_logit_largest = out_logit
            # Other channels
            elif ch!=self.channel_list[1]:
                if self.use_ema:
                    loss = self.distill_loss_fn(out_logit, F.softmax(output_largest, dim=1))
                    with torch.no_grad():
                        if hasattr(self.model_ema.ema, 'module'):
                            self.model_ema.ema.module.set_channel_ratio(ch)
                        else:
                            self.model_ema.ema.set_channel_ratio(ch)
                        guide_output = self.model_ema.ema(input_tensor)
                    guide_list.append((guide_output))
                else:
                    loss = self.train_loss_fn(out_logit, target)
            # 0.25: smallest network
            else:
                if self.use_ema:
                    soft_labels_ = [torch.unsqueeze(guide_list[idx], dim=2) for
                                    idx in range(len(guide_list))]
                    soft_labels_softmax = [F.softmax(i, dim=1) for i in soft_labels_]
                    soft_labels_softmax = torch.cat(soft_labels_softmax, dim=2).mean(dim=2)
                    loss = self.distill_loss_fn(out_logit, soft_labels_softmax)
                else:
                    loss = self.train_loss_fn(out_logit, target)
                loss_smallest = loss.mean()

            # backward
            self.run_backward(loss.mean(), run=True)
        return loss_largest, out_logit_largest

    def recalibrate_bn(self, data_loader):
        if data_loader is None:
            self.logger.log('No bn data loader existed, skip bn recalibration!')
            return
        self.logger.log('Recalibrating BatchNorm statistics ...')
        if self.use_ema:
            model_list = [self.net, self.model_ema.ema]
        else:
            model_list = [self.net]
        for idx, model_ in enumerate(model_list):
            for layer in model_.modules():
                if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm):
                    layer.reset_running_stats()
            model_.train()
            with torch.no_grad():
                for batch_idx, (input_tensor, target) in enumerate(data_loader):
                    for ch in self.channel_list:
                        if hasattr(model_, 'module'):
                            model_.module.set_channel_ratio(ch)
                        else:
                            model_.set_channel_ratio(ch)
                        input_tensor = input_tensor.cuda() #.to(self.device)
                        model_(input_tensor)
            self.logger.log('Finish recalibrating Batchnorm statistics {}. '.format('EMA' if idx==1 else ''))

    def train_policy(self, input_tensor, target):
        # Freeze BN layers of Backbone
        for n, m in self.net.named_modules():
            if isinstance(m, DyChBatchNorm2d):
                m.eval()

        if self.use_meta_net:
            pseudo_net = TSNet(self.cfg, random_init=True)
            if hasattr(self.net, 'module'):
                pseudo_net.load_state_dict(self.net.module.state_dict())
            else:
                pseudo_net.load_state_dict(self.net.state_dict())
            if self.parallel:
                pseudo_net = torch.nn.DataParallel(pseudo_net)
            pseudo_net = pseudo_net.to(self.device)
            # pseudo_net = deepcopy(self.net)
            pseudo_net.train()

            out_logit, _, policy_logit, slim_logit, ch_scores, _ = pseudo_net(input_tensor)
            pseudo_ce_loss = self.train_loss_fn(out_logit, target) + self.train_loss_fn(policy_logit, target)
            pseudo_ce_loss = pseudo_ce_loss.view(-1,1)
            ch_scores = ch_scores.reshape(-1, self.num_segment, len(self.channel_list))  # B, T, K
            ch_lst_tensor = torch.from_numpy(np.array(self.channel_list))**2
            ch_lst_tensor = ch_lst_tensor.unsqueeze(0).unsqueeze(-1).repeat(ch_scores.size(0), 1, 1)  # B,K,1
            ch_lst_tensor = ch_lst_tensor.to(ch_scores.device).float()
            pseudo_flops_loss = torch.bmm(ch_scores, ch_lst_tensor).view(-1,1) # B*T,1
            pseudo_loss_vector = torch.cat((pseudo_ce_loss, pseudo_flops_loss), dim=1)
            pseudo_weight = self.meta_net(pseudo_loss_vector.data)
            sim_loss = self.simliarity_loss_fn(slim_logit, policy_logit)

            pseudo_loss = ((self.alpha+(1-self.alpha)*pseudo_weight) * pseudo_loss_vector).sum(dim=1).mean() + sim_loss
            # pseudo_loss = (pseudo_weight*pseudo_loss_vector).sum(dim=1).mean() + sim_loss

            for param in pseudo_net.parameters():
                param.requires_grad=True
            # pseudo_grads = torch.autograd.grad(pseudo_loss, filter(lambda p: p.requires_grad, pseudo_net.parameters()), create_graph=True)
            pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True, allow_unused=True)
            if hasattr(pseudo_net, 'module'):
                pseudo_net.module.freeze_backbone()
            else:
                pseudo_net.freeze_backbone()

            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), 
                lr=self.cfg.TRAIN.LR,
                momentum=self.cfg.TRAIN.MOMENTUM,
                weight_decay=self.cfg.TRAIN.WD,
                nesterov=self.cfg.TRAIN.NESTEROV)
            # pseudo_optimizer.load_state_dict(self.optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)
            del pseudo_grads

            try:
                meta_inputs, meta_labels, _ = next(self.meta_dataloader_iter)
            except StopIteration:
                self.meta_dataloader_iter = iter(self.meta_dataloader)
                meta_inputs, meta_labels, _ = next(self.meta_dataloader_iter)

            meta_inputs = meta_inputs.to(self.device)
            if self.multi_label:
                meta_labels = meta_labels[:,0].to(self.device)
            else:
                meta_labels = meta_labels.to(self.device)
            _b = meta_labels.shape[0]
            meta_labels_e = meta_labels.view(_b, -1).expand(_b, self.num_segment).reshape(-1)
            out_logit, _, policy_logit, slim_logit, ch_scores, _ = pseudo_net(meta_inputs)
            meta_ce_loss = self.train_loss_fn(out_logit, meta_labels_e) + self.train_loss_fn(policy_logit, meta_labels_e)
            meta_ce_loss = meta_ce_loss.view(-1,1)

            ch_scores = ch_scores.reshape(-1, self.num_segment, len(self.channel_list))  # B, T, K
            ch_lst_tensor = torch.from_numpy(np.array(self.channel_list))**2
            ch_lst_tensor = ch_lst_tensor.unsqueeze(0).unsqueeze(-1).repeat(ch_scores.size(0), 1, 1)  # B,K,1
            ch_lst_tensor = ch_lst_tensor.to(ch_scores.device).float()
            meta_flops_loss = torch.bmm(ch_scores, ch_lst_tensor).view(-1,1) # B*T
            sim_loss = self.simliarity_loss_fn(slim_logit, policy_logit)
            meta_loss = meta_ce_loss.mean() + meta_flops_loss.mean() + sim_loss

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

        out_logit, out_pred, policy_logit, slim_logit, ch_scores, feat = self.net(input_tensor)
        if not self.use_aux_loss:
            ce_loss = self.train_loss_fn(out_logit, target)
        elif not self.use_concat:
            ce_loss = self.train_loss_fn(policy_logit, target)+self.train_loss_fn(slim_logit, target)
            out_pred = policy_logit.view(-1, self.num_segment, policy_logit.size(1)).mean(dim=1)
        else:
            ce_loss = self.train_loss_fn(out_logit, target) + self.train_loss_fn(policy_logit, target) #+ self.train_loss_fn(slim_logit, target)
        
        # ce_loss = ce_loss.view(-1, 1)
        ch_scores = ch_scores.reshape(-1, self.num_segment, len(self.channel_list))  # B, T, K
        ch_lst_tensor = torch.from_numpy(np.array(self.channel_list))**2
        ch_lst_tensor = ch_lst_tensor.unsqueeze(0).unsqueeze(-1).repeat(ch_scores.size(0), 1, 1)  # B,K,1
        ch_lst_tensor = ch_lst_tensor.to(ch_scores.device).float()
        flops_loss = torch.bmm(ch_scores, ch_lst_tensor).view(-1, 1)
        sim_loss = self.simliarity_loss_fn(slim_logit, policy_logit)

        if self.use_meta_net:
            loss_vector = torch.cat((ce_loss, flops_loss), dim=1)
            with torch.no_grad():
                weight = self.meta_net(loss_vector)
                self.logger.add_histogram('ce_loss_weight', weight[:,0].clone().cpu().data.numpy(),self.global_steps)
                self.logger.add_histogram('flops_loss_weight', weight[:,1].clone().cpu().data.numpy(),self.global_steps)
            loss = ((self.alpha+(1-self.alpha)*weight)*loss_vector).sum(dim=1).mean() + sim_loss
            # loss = (weight*loss_vector).sum(dim=1).mean() + sim_loss
        else:
            loss = ce_loss.mean() + 0.7*flops_loss.mean() + sim_loss

        # backward
        self.run_backward(loss, run=True)
        return loss, out_pred #, (ce_loss.view(-1, self.num_segment).mean(dim=1), flops_loss.view(-1, self.num_segment).mean(dim=1), sim_loss)
  
    def predict_and_eval_in_val(self, val_loader, metrics):
        if self.train_stage=='supernet':
            for ch in self.channel_list:
                self.validate_supernet(val_loader, metrics, ch)
        else:
            self.validate_policy(val_loader, metrics)
    
    def validate_supernet(self, val_loader, metrics, ch):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        prec1_m = AverageMeter()
        prec5_m = AverageMeter()

        ema_losses_m = AverageMeter()
        ema_prec1_m = AverageMeter()
        ema_prec5_m = AverageMeter()

        if self.multi_label:
            all_target = []
            all_results = []
            all_ema_res = []
        if hasattr(self.net, 'module'):
            self.net.module.set_channel_ratio(ch)
            self.model_ema.ema.module.set_channel_ratio(ch)
        else:
            self.net.set_channel_ratio(ch)
            self.model_ema.ema.set_channel_ratio(ch)

        if self.show_process_bar:
            nbar = tqdm(total=len(val_loader))
        
        end = time.time()
        for batch_idx, (input_tensor, target, _) in enumerate(val_loader):
            if self.show_process_bar:
                nbar.update(1)
            input_tensor = input_tensor.to(self.device)
            if self.multi_label:
                all_target.append(target)
                if self.cfg.DATASET.NAME=='anet':
                    target = target[:,0]
            target = target.to(self.device)
        
            out_logit = self.net(input_tensor)
            loss = self.validate_loss_fn(out_logit, target)
            losses_m.update(loss.item())

            if self.multi_label:
                all_results.append(out_logit)
            else:
                prec1, prec5 = accuracy(out_logit.data, target, topk=(1, 5))
                prec1_m.update(prec1.item(), out_logit.size(0))
                prec5_m.update(prec5.item(), out_logit.size(0))

            ema_out = self.model_ema.ema(input_tensor)
            ema_loss = self.validate_loss_fn(ema_out, target)
            ema_losses_m.update(ema_loss.item())

            if self.multi_label:
                all_ema_res.append(ema_out)
            else:
                ema_prec1, ema_prec5 = accuracy(ema_out.data, target, topk=(1,5))
                ema_prec1_m.update(ema_prec1.item(), ema_out.size(0))
                ema_prec5_m.update(ema_prec5.item(), ema_out.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
        
        if self.show_process_bar:
            nbar.close()

        metrics['channel_%.2f_loss'%ch] = losses_m.avg
        metrics['ema_channel_%.2f_loss'%ch] = ema_losses_m.avg
        metrics['channel_%.2f_batch_time'%ch] = batch_time_m.avg
        if self.multi_label:
            if self.cfg.DATASET.NAME == 'fcvid':
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_target, 0).cpu(), is_soft=True)
                ema_mAP, _ = cal_map(torch.cat(all_ema_res, 0).cpu(), torch.cat(all_target, 0).cpu(), is_soft=True)
            else:
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_target, 0)[:, 0:1].cpu())
                ema_mAP, _ = cal_map(torch.cat(all_ema_res, 0).cpu(), torch.cat(all_target, 0)[:, 0:1].cpu())
            
            metrics['channel_%.2f_map'%ch] = mAP
            if metrics['channel_%.2f_map'%ch] > self.best_metrics:
                self.best_metrics = metrics['channel_%.2f_map'%ch]
                self.is_best = True
            metrics['ema_channel_%.2f_map'%ch] = ema_mAP
            if metrics['ema_channel_%.2f_map'%ch] > self.best_metrics:
                self.best_metrics = metrics['ema_channel_%.2f_map'%ch]
                self.is_best = True
        else:
            metrics['channel_%.2f_top1'%ch] = prec1_m.avg
            metrics['channel_%.2f_top5'%ch] = prec5_m.avg
            metrics['ema_channel_%.2f_top1'%ch] = ema_prec1_m.avg
            metrics['ema_channel_%.2f_top5'%ch] = ema_prec5_m.avg

            if metrics['channel_%.2f_top1'%ch] > self.best_metrics:
                self.best_metrics = metrics['channel_%.2f_top1'%ch]
                self.is_best = True
            if metrics['ema_channel_%.2f_top1'%ch] > self.best_metrics:
                self.best_metrics = metrics['ema_channel_%.2f_top1'%ch]
                self.is_best = True
        
        for k,v in metrics.items():
            self.logger.add_scalar(k, v, self.global_steps)

    def validate_policy(self, val_loader, metrics):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        prec1_m = AverageMeter()
        prec5_m = AverageMeter()
        ch_ratio_m = AverageMeter()

        # running_flops = add_flops(self._net)
        if self.multi_label:
            all_target = []
            all_results = []
        if self.show_process_bar:
            nbar = tqdm(total=len(val_loader))

        end = time.time()
        for batch_idx, (input_tensor, target, _) in enumerate(val_loader):
            if self.show_process_bar:
                nbar.update(1)
            input_tensor = input_tensor.to(self.device)
            if self.multi_label:
                all_target.append(target)
                target = target[:,0].to(self.device)
            else:
                target = target.to(self.device)
            _b = target.shape[0]
            target_e = target.view(_b, -1).expand(_b, self.num_segment).reshape(-1)
        
            out_logit, out_pred, policy_logit, slim_logit, ch_scores, _ = self.net(input_tensor)
            if not self.use_concat:
                out_logit = policy_logit
                out_pred = policy_logit.view(-1, self.num_segment, policy_logit.size(1)).mean(dim=1)
            if self.multi_label:
                all_results.append(out_pred)

            loss = self.validate_loss_fn(out_logit, target_e)
            prec1, prec5 = accuracy(out_pred.data, target, topk=(1, 5))
            losses_m.update(loss.item())
            prec1_m.update(prec1.item(), out_logit.size(0))
            prec5_m.update(prec5.item(), out_logit.size(0))

            ch_scores = ch_scores.reshape(-1, self.num_segment, len(self.channel_list)) # B, T, K
            ch_lst_tensor = torch.from_numpy(np.array(self.channel_list))**2
            ch_lst_tensor = ch_lst_tensor.unsqueeze(0).unsqueeze(-1).repeat(ch_scores.size(0), 1, 1)  # B,K,1
            ch_lst_tensor = ch_lst_tensor.to(ch_scores.device).float()
            ch_ratio = torch.bmm(ch_scores, ch_lst_tensor).mean()
            ch_ratio_m.update(ch_ratio.item(), out_logit.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
        
        if self.show_process_bar:
            nbar.close()
        
        metrics['loss'] = losses_m.avg
        metrics['top1'] = prec1_m.avg
        metrics['top5'] = prec5_m.avg
        metrics['dynamic_ch_ratio'] = ch_ratio_m.avg

        if self.multi_label:
            if self.cfg.DATASET.NAME=='fcvid':
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_target, 0).cpu())
            else:
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_target, 0)[:,0:1].cpu())
            metrics['dynamic_map'] = mAP
            if metrics['dynamic_map'] > self.best_metrics:
                self.best_metrics = metrics['dynamic_map']
                self.is_best = True
        else:
            if metrics['top1'] > self.best_metrics:
                self.best_metrics = metrics['top1']
                self.is_best = True
        
        for k,v in metrics.items():
            self.logger.add_scalar(k, v, self.global_steps)

    def predict_in_tst(self, test_reader):
        batch_time_m = AverageMeter()
        prec1_m = AverageMeter()
        prec5_m = AverageMeter()
        ch_ratio_m = AverageMeter()
        flops_m = AverageMeter()

        # fr_acc = [AverageMeter() for _ in range(self.num_segment)]
        vid_id_dict = {}
        pri_feat = []
        sup_feat = []

        if self.multi_label:
            all_target = []
            all_results = []
            fr_res = [[] for _ in range(self.num_segment)]
            
        if self.show_process_bar:
            nbar = tqdm(total=len(test_reader))
        self.net.apply(lambda m: add_mac_hooks(m))
        first_time=True

        # Test dynamic network
        for input_tensor, target, vid_ids in test_reader:
            if self.show_process_bar:
                nbar.update(1)
            input_tensor = input_tensor.to(self.device)
            if self.multi_label:
                all_target.append(target)
                target = target[:,0].to(self.device)
            else:
                target = target.to(self.device)
            
            _b = target.size(0)
            if first_time:
                print(_b)
                first_time=False
            out_logit, out_pred, ch_scores, ch_idx, policy_feat, slim_feat = self.net(input_tensor)
            if self.multi_label:
                all_results.append(out_pred)
                for i in range(self.num_segment):
                    fr_pred = out_logit.view(_b, self.num_segment,-1)[:,i,:]
                    fr_res[i].append(fr_pred)
            running_flops = add_flops(self.net)
            #print(running_flops)
            flops_m.update(running_flops*self.num_segment, _b)

            prec1, prec5 = accuracy(out_pred.data, target, topk=(1, 5))
            prec1_m.update(prec1.item(), _b)
            prec5_m.update(prec5.item(), _b)

            pri_feat.append(policy_feat)
            sup_feat.append(slim_feat)

            ch_scores = ch_scores.reshape(-1, self.num_segment, len(self.channel_list)) # B, T, K
            ch_lst_tensor = torch.from_numpy(np.array(self.channel_list))**2
            ch_lst_tensor = ch_lst_tensor.unsqueeze(0).unsqueeze(-1).repeat(ch_scores.size(0), 1, 1)  # B,K,1
            ch_lst_tensor = ch_lst_tensor.to(ch_scores.device).float()
            ch_ratio = torch.bmm(ch_scores, ch_lst_tensor).mean()

            ch_ratio_m.update(ch_ratio.item(), _b)
            ch_idx = ch_idx.detach().cpu().numpy()
            for i, v_id in enumerate(vid_ids):
                vid_id_dict[v_id] = ch_idx[i*self.num_segment:(i+1)*self.num_segment].tolist()
        
        if self.show_process_bar:
            nbar.close()
        
        mAP = 0
        if self.multi_label:
            if self.cfg.DATASET.NAME == 'fcvid':
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_target, 0).cpu())
            else:
                for i, res in enumerate(fr_res):
                    f_ap, _ = cal_map(torch.cat(res, 0).cpu(), torch.cat(all_target, 0)[:, 0:1].cpu())
                    self.logger.log('mAP@f%d: %.2f, '%(i, f_ap))
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_target, 0)[:, 0:1].cpu())

        self.logger.log('map: %.2f, prec1: %.2f, prec5: %.2f, dynamic flops ratio: %.5f'%(mAP, prec1_m.avg, prec5_m.avg, ch_ratio_m.avg))
        
        np.save('./{}-primary-feat.npy'.format(self.cfg.DATASET.NAME), torch.cat(pri_feat, dim=0).detach().cpu().numpy())
        np.save('./{}-suppliment-feat.npy'.format(self.cfg.DATASET.NAME), torch.cat(sup_feat, dim=0).detach().cpu().numpy())
        json.dump(vid_id_dict, open('./{}_video_list_test.json'.format(self.cfg.DATASET.NAME),'w'), indent=2)

    def run_backward(self, loss, run=False):
        if run:
            loss = loss / self.cfg.TRAIN.ACCUM_N_BS  # loss regularization
            if self.cfg.TRAIN.USE_APEX:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
