import os
from abc import abstractmethod
from statistics import mean

import time
import torch
import pandas as pd
from tqdm.auto import tqdm
from numpy import inf
from .logger import FileLogger, WandbLogger


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.steps = self.args.steps
        self.eval_steps = self.args.eval_steps

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val/' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_step = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}
        self.logger = (
            WandbLogger(args.save_dir, args.visual_extractor)
            if self.args.logger == 'wandb'
            else FileLogger(args.save_dir, args.visual_extractor)
        )

    @abstractmethod
    def _train_step(self):
        raise NotImplementedError

    @abstractmethod
    def _val_step(self):
        raise NotImplementedError

    @abstractmethod
    def _test_epoch(self):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        step = self.start_step
        train_loss = []

        pbar = tqdm(total=self.steps)
        while step < self.steps:
            self.model.train()
            for train_images_id, train_images, train_reports_ids, train_reports_masks in self.train_dataloader:
                step_loss = self._train_step(train_images, train_reports_ids, train_reports_masks)
                train_loss.append(step_loss)
                step += 1
                if step % self.eval_steps == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_gts, val_res = [], []
                        for val_images_id, val_images, val_reports_ids, val_reports_masks in self.val_dataloader:
                            reports, ground_truths = self._val_step(val_images, val_reports_ids, val_reports_masks)
                            val_res.extend(reports)
                            val_gts.extend(ground_truths)
                        val_met = self.metric_ftns(
                            {i: [gt] for i, gt in enumerate(val_gts)},
                            {i: [re] for i, re in enumerate(val_res)}
                        )
                        log = {
                            'step': step,
                            'train/loss': mean(train_loss)
                        }
                        log.update(**{'val/' + k: v for k, v in val_met.items()})
                        self.logger.log_step(log)

                        self._record_best(log)

                        # print logged informations to the screen
                        for key, value in log.items():
                            print('\t{:15s}: {}'.format(str(key), value))

                        # evaluate model performance according to configured metric, save best checkpoint as model_best
                        best = False
                        if self.mnt_mode != 'off':
                            try:
                                # check whether model performance improved or not, according to specified metric(mnt_metric)
                                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                            except KeyError:
                                print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                                    self.mnt_metric))
                                self.mnt_mode = 'off'
                                improved = False

                            if improved:
                                self.mnt_best = log[self.mnt_metric]
                                not_improved_count = 0
                                best = True
                            else:
                                not_improved_count += 1

                        train_loss = []
                        pbar.update(1)
                        self._save_checkpoint(step, save_best=best)
                        if not_improved_count > self.early_stop:
                            print("Validation performance didn\'t improve for {} steps. " "Training stops.".format(
                                self.early_stop))
                            break
                        if step > self.steps:
                            break
            self.lr_scheduler.step()
        self._print_best()
        self._print_best_to_file()
        filename = os.path.join(self.checkpoint_dir, 'model_best.pth')
        state_dict = torch.load(filename)['state_dict']
        self.model.load_state_dict(state_dict)

    def test(self):
        log, test_ids, test_res, test_gts = self._test_epoch()
                # print logged informations to the screen
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

        record_table = pd.DataFrame({
            'image_id': test_ids,
            'ground_truth': test_gts,
            'inference': test_res
        })
        self.logger.log_table(record_table)
        self.logger.log_model(log, 'model_best.pth')

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'

        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, step, save_best=False):
        state = {
            'step': step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            print("Saving current best: model_best.pth ...")
            torch.save(state, best_path)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_step = checkpoint['step'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from step {}".format(self.start_step))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_step(self, images, reports_ids, reports_masks):
        images, reports_ids, reports_masks = (
            images.to(self.device),
            reports_ids.to(self.device),
            reports_masks.to(self.device)
        )
        output = self.model(images, reports_ids, mode='train')
        loss = self.criterion(output, reports_ids, reports_masks)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
        self.optimizer.step()
        return loss.item()

    def _val_step(self,  images, reports_ids, reports_masks):
        images, reports_ids, reports_masks = (
            images.to(self.device),
            reports_ids.to(self.device),
            reports_masks.to(self.device)
        )
        output = self.model(images, mode='sample')
        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        return reports, ground_truths

    def _test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            test_ids, test_gts, test_res = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_ids.extend(images_id)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

        return test_met, test_ids, test_res, test_gts
