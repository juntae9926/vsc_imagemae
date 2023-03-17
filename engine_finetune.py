# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, positives) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        positives = positives.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # outputs = model(samples)
            # loss = criterion(outputs, targets)
            _, r_loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            latent_original, _, _, _ = model(samples, mask_ratio=0.0)
            latent_positive, _, _, _ = model(positives, mask_ratio=0.0)
        c_loss = criterion(latent_original, latent_positive)

        total_loss = r_loss + c_loss
        if not math.isfinite(total_loss):
            print("Total Loss is {}, stopping training".format(total_loss))
            sys.exit(1)

        total_loss /= accum_iter
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # metric_logger.update(loss=r_loss)
        # metric_logger.update(loss=c_loss)
        metric_logger.update(loss=total_loss)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=max_lr)

        total_loss_reduce = misc.all_reduce_mean(total_loss)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', total_loss_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args):
    #criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    # header = 'Test:'
    header = '[Validation]'
    # switch to evaluation mode
    model.eval()
    total_loss = 0.0

    # for batch_idx, val_dict in enumerate(data_loader):
    for batch_idx, val_dict in enumerate(metric_logger.log_every(data_loader, 3, header)):
        q_label = val_dict['q_name']
        q_frame = val_dict['q_img']
        r_label = val_dict['r_name']
        r_frame = val_dict['r_img']

        q_frame = q_frame.to(device, non_blocking=True)
        r_frame = r_frame.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            _, r_loss, _, _ = model(q_frame, mask_ratio=args.mask_ratio)
            latent_q_frame, _, _, _ = model(q_frame, mask_ratio=0.0)
            latent_r_frame, _, _, _ = model(r_frame, mask_ratio=0.0)
        c_loss = criterion(latent_q_frame, latent_r_frame)

        total_loss = r_loss + c_loss
        metric_logger.update(loss=total_loss.item())

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # return total_loss.item()
    return total_loss