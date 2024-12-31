"""
General training, validation and testing schema.

Reorganized from: https://github.com/marshimarocj/conv_rnn_trn/
"""
import copy
from utils.helpers import *
from utils import AverageMeter
import traceback
import random
from tqdm import tqdm
try:
    from apex import amp
except ImportError:
    pass
from torch.utils.data import DataLoader


class BaseTrainer(object):
    def __init__(self, net, cfg, logger=None):
        self._net = net
        self._cfg = cfg
        self.logger = logger
        self.modality = cfg.DATASET.MODALITY
        self.show_process_bar = cfg.SHOW_BAR
        self._enable_pbn = cfg.TRAIN.PARTIAL_BN
        self.parallel = True if len(cfg.GPUS) > 1 else False
        self.optimizer = get_optimizer(self.cfg, self._net, policies=self.get_optim_policies())
        self.checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, '.'.join([cfg.SNAPSHOT_PREF, cfg.MODEL.NAME]))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.best_metrics = 0  # initialized best metrics
        self.best_loss = 1000  # initialized best loss
        self.is_best = False
        self.use_ema= False
        self.global_steps = 0
        self.train_metrics = {}
        self.device = 'cuda' if torch.cuda.device_count()>=1 else torch.device('cpu')

        if self.cfg.TRAIN.USE_APEX:
            self._net, self.optimizer = amp.initialize(self._net, self.optimizer, opt_level='O1')

    @property
    def cfg(self):
        return self._cfg

    @property
    def net(self):
        return self._net

    def init_op(self):
        '''
        Implement initialization operations.
        '''
        raise NotImplementedError("""please customize init_op""")

    def feed_data_and_run_loss(self, data):
        """
        return loss value
        """
        raise NotImplementedError("""please customize feed_data_and_run_loss""")

    def predict_and_eval_in_val(self, val_loader, metrics):
        """
        add eval result to metrics dictionary, key is metric name, val is metric value
        """
        raise NotImplementedError("""please customize predict_and_eval_in_val""")

    def predict_in_tst(self, tst_loader):
        """
        write predict result to predict_file
        """
        raise NotImplementedError("""please customize predict_in_tst""")

    def adjust_learning_rate(self, optimizer, epoch, lr_type, lr_steps):
        """
        adjust learning rate by customer
        """
        raise NotImplementedError("""please customize adjust_learning_rate""")

    def get_optim_policies(self):
        return None

    def get_lr_schedule(self, start_epoch):
        warm_up_epoch = 0
        if self.cfg.TRAIN.WARM_UP:
            warm_up_epoch = self.cfg.TRAIN.WARM_UP_EPOCHS

        if self.cfg.TRAIN.LR_SCHEDULER == 'cosine':
            lr_lamda = lambda epoch: (epoch-self.cfg.TRAIN.BEGIN_EPOCH)/warm_up_epoch if epoch <= warm_up_epoch \
                else 0.5*(1 + math.cos(math.pi*(epoch-warm_up_epoch)/(self.cfg.TRAIN.END_EPOCH-warm_up_epoch)))
        else:
            # Default: MultiStepLR
            lr_lamda = lambda epoch: (epoch-self.cfg.TRAIN.BEGIN_EPOCH) / warm_up_epoch if epoch < warm_up_epoch \
                else self.cfg.TRAIN.LR_FACTOR ** len([m for m in self.cfg.TRAIN.LR_STEP if m <= epoch])

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                      lr_lambda=lr_lamda,
                                                      last_epoch=start_epoch - 1)
        if self.cfg.TRAIN.LR_SCHEDULER == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                   min_lr=1e-7,
                                                                   patience=self.cfg.TRAIN.PATIENCE,
                                                                   factor=self.cfg.TRAIN.LR_FACTOR,
                                                                   verbose=True)
        return scheduler

    def run_backward(self, loss):
        loss = loss / self.cfg.TRAIN.ACCUM_N_BS  # loss regularization
        if self.cfg.TRAIN.USE_APEX:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def end_ops(self, data_loader):
        """
        operations after training the model
        """
        pass

    def gen_meta_dataset(self, dataset):
        # if hasattr(self, 'sample_ind'):
        #     print('Generating meta dataset')
        #     assert len(self.sample_ind) == len(self.ce_score)
        #     assert len(self.ce_score) == len(self.flop_score)
        #     _, first_ind = torch.sort(torch.cat(self.ce_score))
        #     all_n = len(first_ind)
        #     flop_score = torch.cat(self.flop_score)[first_ind[:min(self.meta_num*2, all_n)]]

        #     _, sec_ind = torch.sort(flop_score)
        #     org_ind = first_ind[sec_ind[:min(self.meta_num, all_n)]]
        #     meta_index = torch.cat(self.sample_ind)[org_ind]
            
        #     meta_dataset = copy.deepcopy(dataset)
        #     meta_dataset.anno = [meta_dataset.anno[x] for x in list(meta_index.numpy())]
        # else:
        #     meta_dataset = copy.deepcopy(dataset)
        #     meta_dataset.anno = meta_dataset.anno[:self.meta_num]
        # self.sample_ind, self.ce_score, self.flop_score = [],[],[]
        
        meta_dataset = copy.deepcopy(dataset)
        ind = random.sample(list(range(len(meta_dataset))), self.meta_num)
        meta_dataset.anno = [meta_dataset.anno[x] for x in ind]
        self.meta_dataloader = DataLoader(meta_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True, pin_memory=True)
        self.meta_dataloader_iter = iter(self.meta_dataloader)

    def train_one_epoch(self, train_loader, epoch):
        total_step = len(train_loader)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        if self.use_meta_net:
            self.alpha = math.exp(-0.005*(epoch-self.cfg.TRAIN.BEGIN_EPOCH)**2)
            self.gen_meta_dataset(train_loader.dataset)
            
        # update parameters for each batch
        end_time = time.time()
        last_idx = len(train_loader) - 1
        if self.show_process_bar:
            pbar = tqdm(total=len(train_loader))
        for num_step, data in enumerate(train_loader):
            last_batch = num_step == last_idx
            if self.show_process_bar:
                pbar.update(1)
            # measuring data loading time
            data_time.update(time.time() - end_time)

            # forward
            loss = self.feed_data_and_run_loss(data)

            # backward
            self.run_backward(loss)

            # accumulate gradient and update weights
            if last_batch or num_step % self.cfg.TRAIN.ACCUM_N_BS==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.use_ema and hasattr(self, 'model_ema'):
                    self.model_ema.update(self._net)
            
            # record loss and parameters
            losses.update(loss.item(), data[0].size(0))
            self.logger.add_scalar('train_loss', losses.val, self.global_steps)
            for name, param in self._net.named_parameters():
                for item in self.cfg.MODEL.HIST_SHOW_NAME:
                    if item in name:
                        self.logger.add_histogram(self.cfg.MODEL.NAME+'_'+name,
                                                  param.clone().cpu().data.numpy(),
                                                  self.global_steps)
            self.global_steps += 1

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # save epoch
            if last_batch or self.global_steps % self.cfg.SAVE_FREQ == 0:
                save_path = os.path.join(self.checkpoint_dir, 'epoch_latest.pth')
                save_checkpoint(checkpoint_file=save_path,
                                net=self._net,
                                epoch=epoch,
                                gs=self.global_steps,
                                optim=self.optimizer,
                                is_parallel=self.parallel)
                self.logger.log('saving checkpoint to {}'.format(save_path))

                if hasattr(self, 'model_ema'):
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_ema_latest.pth')
                    save_checkpoint(save_path,
                                    self.model_ema.ema,
                                    epoch=epoch,
                                    gs=self.global_steps,
                                    optim=self.optimizer,
                                    is_parallel=self.parallel)
                    self.logger.log('saving checkpoint to {}'.format(save_path))

            # display training info
            if last_batch or num_step % self.cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t ' \
                      'Speed {speed:.1f} samples/s\t Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t ' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f}))'.format(
                    epoch, num_step, total_step, batch_time=batch_time,
                    speed=data[0].size(0)/batch_time.val,
                    data_time=data_time,
                    loss=losses)
                self.logger.log(msg)

        if self.show_process_bar:
            pbar.close()
        return losses.avg

    def _validation(self, val_loader):
        metrics = {}
        with torch.no_grad():
            self.predict_and_eval_in_val(val_loader, metrics)
        return metrics

    def train(self, data_loader, **kwarg):
        start_epoch = self.cfg.TRAIN.BEGIN_EPOCH
        if self.cfg.AUTO_RESUME:
            try:
                which_epoch = kwarg['which_epoch']
                if which_epoch is None:
                    which_epoch = 'latest'
                    have_key = False
                else:
                    have_key = True
            except KeyError:
                which_epoch = 'latest'
                have_key = False
            checkpoint_file = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(which_epoch))
            if os.path.exists(checkpoint_file):
                self.logger.log('=> loading checkpoint {}'.format(checkpoint_file))
                w_dict, chech_info = load_checkpoint(checkpoint_file, self.parallel)
                self._net.load_state_dict(w_dict)
                self.optimizer.load_state_dict(chech_info['optimizer'])
                start_epoch = chech_info['epoch']+1
                self.global_steps = chech_info['global_step']
            else:
                if have_key:
                    raise(ValueError, 'checkpoint file of epoch {} not existed!'.format(which_epoch))

        self.net.train()
        # round 0, just for quick check
        if self.cfg.USE_VAL and self.cfg.TEST.INIT_VAL:
            self.logger.log('Initial validation')
            self._net.eval()
            try:
                metrics = self._validation(kwarg['val_loader'])
            except KeyError:
                raise (KeyError, 'No validation data loader defined')
            for key in metrics:
                self.logger.log('{}:{}'.format(key, metrics[key]))
            self._net.train()

        lr_scheduler = self.get_lr_schedule(start_epoch)

        for epoch in range(start_epoch, self.cfg.TRAIN.END_EPOCH+1):
            start_time = time.time()
            # train for one epoch
            self.train_one_epoch(data_loader, epoch=epoch)

            # validate
            if self.cfg.USE_VAL and epoch>=self.cfg.TEST.START_VAL_EPOCH and epoch % self.cfg.TEST.VAL_FREQ == 0:
                self.logger.log('Validating...')
                self.net.eval()
                try:
                    metrics = self._validation(kwarg['val_loader'])
                except KeyError:
                    raise (KeyError, 'No validation data loader defined')
                for key in metrics:
                    self.logger.log('{}:{}'.format(key, metrics[key]))
                self.logger.log('Best metric: {}'.format(self.best_metrics))
                if self.is_best:
                    save_path = os.path.join(self.checkpoint_dir, 'best_model.pth'.format(epoch))
                    save_checkpoint(save_path,
                                    self._net,
                                    epoch=epoch,
                                    gs=self.global_steps,
                                    optim=self.optimizer,
                                    is_parallel=self.parallel)
                    if hasattr(self, 'model_ema'):
                        save_path = os.path.join(self.checkpoint_dir, 'best_ema_model.pth'.format(epoch))
                        save_checkpoint(save_path,
                                    self.model_ema.ema,
                                    epoch=epoch,
                                    gs=self.global_steps,
                                    optim=self.optimizer,
                                    is_parallel=self.parallel)

                    self.is_best = False
                self.net.train()

            # Print info: epoch, lr, time, metrics
            lr_list = [str(param_group['lr']) for param_group in self.optimizer.param_groups]
            self.logger.log('=>learning rate: '+' '.join(lr_list))
            self.logger.log('=>epoch: ({}/{}), cost {:.3f}s'.format(
                epoch, self.cfg.TRAIN.END_EPOCH, time.time() - start_time))
            for k, v in self.train_metrics.items():
                self.logger.log('=>' + k + ':' + str(v.avg))
            try:
                torch.cuda.empty_cache()
            except:
                pass

            # update learning rate
            try:
                self.adjust_learning_rate(self.optimizer, epoch, lr_type=self.cfg.TRAIN.LR_SCHEDULER, lr_steps=self.cfg.TRAIN.LR_STEP)
            except NotImplementedError:
                if self.cfg.TRAIN.LR_SCHEDULER == 'plateau':
                    # TODO: process conflict with INIT_VAL
                    lr_scheduler.step(metrics['loss'])
                    # try:
                    #     lr_scheduler.step(metrics['loss'])
                    # except :
                    #     self.logger.log('Please set INIT_VAL as true when using plateau lr_scheduler.')
                else:
                    lr_scheduler.step()

        self.end_ops(data_loader)
        self.logger.writer.close()

    def test(self, test_reader, weight_file):
        state_dict, _, = load_checkpoint(weight_file, self.parallel)
        self.logger.log('=> loading checkpoint {}'.format(weight_file))
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()
        self.logger.log('Testing...')
        with torch.no_grad():
            self.predict_in_tst(test_reader)
