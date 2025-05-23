import os
import time
from time import ctime
import logging
import multiprocessing.util

# from sklearnex import patch_sklearn
# patch_sklearn()

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from importlib import import_module
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from survival_loss import NLLDeepSurvLoss

# from torcheval.metrics.aggregation.auc import AUC
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score

import shutil
import warnings
import builtins
import random
import math
import sys
from shutil import copyfile

from dataset import ImageDataset, BalancedDistributedSampler
from model.utils import get_model
from utils import AverageMeter, calculate_accuracy, save_checkpoint, adjust_learning_rate, ProgressMeter

from config import create_arg_parser


def main():
    args = create_arg_parser()
    
    if args.gpu is not None:
        args.gpu = [int(i) for i in args.gpu.split(',')]
    print(args.gpu)
    # 创建保存路径
    train_start_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    
    # CT(1) pathology(2) CI(3)
    if args.modality == ['CT', 'pathology', 'CI']:
        modality_used = '123'
        model_name = args.model_CT + '-' + args.model_pathology + '-' + args.model_CI + '(' + args.aggregator + ')'
    elif args.modality == ['CT']:
        modality_used = '1'
        model_name = args.model_CT + '(' + args.aggregator + ')'
    elif args.modality == ['pathology', 'CI']:
        modality_used = '23'
        model_name = args.model_pathology + '-' + args.model_CI + '(' + args.aggregator + ')'
    elif args.modality == ['CT', 'CI']:
        modality_used = '13'
        model_name = args.model_CT + '-' + args.model_CI + '(' + args.aggregator + ')'
    elif args.modality == ['pathology']:
        modality_used = '2'
        model_name = args.model_pathology + '(' + args.aggregator + ')'
    elif args.modality == ['CI']:
        modality_used = '3'
        model_name = args.model_CI + '(' + args.aggregator + ')'

    # 简化保存路径
    save_dir = 'results/SavedModels/modality(%s)/%s/[%d]%s' % (
        modality_used,
        model_name,
        args.val_fold,
        train_start_time
    )
    
    os.makedirs(save_dir, exist_ok=True)
        
    if args.distributed:
        args.ngpus_per_node = len(args.gpu)
        args.world_size = args.ngpus_per_node
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, 
                args=(args.ngpus_per_node, args, save_dir))  # pass save_dir
    else:
        main_worker(args.gpu, 1, args, save_dir)  # pass save_dir

def main_worker(gpu, ngpus_per_node, args, save_dir):  # receive save_dir
    args.gpu = gpu
    
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                              init_method=args.dist_url,
                              world_size=args.world_size,
                              rank=args.rank)
    
    writer = SummaryWriter(log_dir=save_dir.replace('results/SavedModels', 'runs'))

    if args.distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training.".format(args.gpu))

    # model = get_model(args, weights=args.pretrained_weights)
    model = get_model(args)
    if args.distributed:
        if args.gpu is not None:
            print(args.gpu)
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            generator = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            generator = DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDAtaParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDAtaParallel is supported.")
    
    
    cudnn.benchmark = True

    bce_criterion = torch.nn.MSELoss().cuda(args.gpu)
    cox_criterion = NLLDeepSurvLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(generator.parameters(),
                                    lr=args.lr,
                                    betas=(args.b1, args.b2),
                                    weight_decay=1e-7)
    

    train_dataset = ImageDataset(args, mode='train')
    valid_dataset = ImageDataset(args, mode='valid')
    if args.distributed:
        print("args.world_size: ", args.world_size)
        train_sampler = BalancedDistributedSampler(
            dataset=train_dataset,
            num_replicas=args.world_size,
            rank=args.rank
        )
    else:
        print("args.batch_size: ", args.batch_size)
        train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)

    dataloader_train = DataLoader(
        train_dataset, 
        batch_size=2,
        batch_sampler=train_sampler if not args.distributed else None,
        sampler=train_sampler if args.distributed else None,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    dataloader_valid = DataLoader(valid_dataset, batch_size=1, shuffle=False,num_workers=args.num_workers, pin_memory=True)
    
    valid_c_index_best = 0
    for epoch in range(args.n_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        print(save_dir)
        print("---------------------------------------------------------------------------------------------------------------------------")
        train(dataloader_train, generator, bce_criterion, cox_criterion, optimizer, epoch, args, writer)
        print("---------------------------------------------------------------------------------------------------------------------------")
        _, c_index = valid(dataloader_valid, generator, bce_criterion, cox_criterion, optimizer, epoch, args, writer)
        print("---------------------------------------------------------------------------------------------------------------------------")

        if not args.distributed or (
            args.distributed and args.rank % ngpus_per_node == 0
        ):
            print('c_index: ', c_index)
            is_best = c_index > valid_c_index_best
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=is_best,
                    save_dir = save_dir,
                    filename="checkpoint_{:04d}.pth.tar".format(epoch),
                )
                valid_c_index_best = c_index
            torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, save_dir + '/checkpoint_last.pth.tar')

def train(dataloader_train, generator, bce_criterion, cox_criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(dataloader_train),
        [batch_time, data_time, losses],
        prefix="Train Epoch: [{}]".format(epoch),
    )
    
    risk_scores_all = []
    survival_times_all = []
    events_all = []
    
    ce_criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
    generator.train()

    end = time.time()
    for i, train_data_dict in enumerate(dataloader_train):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if ('CT' in args.modality):
                train_input_CT_pre = train_data_dict['input_CT_pre'].cuda(non_blocking=True)
                train_input_CT_post = train_data_dict['input_CT_post'].cuda(non_blocking=True)
                train_label_CT_pre = train_data_dict['label_CT_pre'].cuda(non_blocking=True)
                train_label_CT_post = train_data_dict['label_CT_post'].cuda(non_blocking=True)
                bs, num_crops, c, h, w = train_input_CT_pre.shape
                train_input_CT_pre = train_input_CT_pre.view(-1,*train_input_CT_pre.shape[-3:]).unsqueeze(1)
                train_input_CT_post = train_input_CT_post.view(-1,*train_input_CT_post.shape[-3:]).unsqueeze(1)
                train_label_CT_pre = train_label_CT_pre.view(-1,*train_label_CT_pre.shape[-3:]).unsqueeze(1)
                train_label_CT_post = train_label_CT_post.view(-1,*train_label_CT_post.shape[-3:]).unsqueeze(1)
            survival_time = train_data_dict['survival_time'].cuda(non_blocking=True)
            event_indicator = train_data_dict['event_indicator'].cuda(non_blocking=True)

            survival_time = survival_time.view(-1).unsqueeze(1)
            event_indicator = event_indicator.view(-1).unsqueeze(1)
            
        # print(train_input_CT_pre.shape)
        risk_score, survival_pred, _, _ = generator(
            [train_input_CT_pre, train_input_CT_post], 
            [train_label_CT_pre, train_label_CT_post],
        )

        five_year_survival_label = torch.tensor(
            [[1.0] if (st >= 60.0) else [0.0] for st in survival_time],
            device=risk_score.device
        )
        cox_loss = cox_criterion(risk_score, survival_time, event_indicator) + 0.1*ce_criterion(survival_pred, five_year_survival_label)
        # surv_loss = ce_criterion(survival_pred, five_year_survival_label)
        survival_loss = cox_loss
        # print(cox_loss)
        # print(surv_loss)
        # if event_indicator == 1:
        #     survival_loss += bce_criterion(survival_pred, survival_time)
        train_loss = survival_loss

        risk_scores_all.append(risk_score.detach().cpu())
        survival_times_all.append(survival_time.detach().cpu())
        events_all.append(event_indicator.detach().cpu())

        losses.update(train_loss.item(), risk_score.size(0))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    
    risk_scores_all = torch.cat(risk_scores_all).numpy()
    survival_times_all = torch.cat(survival_times_all).numpy()
    events_all = torch.cat(events_all).numpy()
    c_index = calculate_c_index(risk_scores_all, survival_times_all, events_all)
    
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/c_index', c_index, epoch)

def valid(dataloader_valid, generator, bce_criterion, cox_criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(dataloader_valid),
        [batch_time, data_time, losses],
        prefix="Valid Epoch: [{}]".format(epoch),
    )
    
    risk_scores_all = []
    survival_times_all = []
    events_all = []

    
    # ce_criterion = BinaryFocalLoss(alpha=0.75, gamma=2).cuda(args.gpu)
    ce_criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
    
    with torch.no_grad():
        generator.eval()

        end = time.time()
        for i, valid_data_dict in enumerate(dataloader_valid):
            data_time.update(time.time() - end)

            if args.gpu is not None:
                if ('CT' in args.modality):
                    valid_input_CT_pre = valid_data_dict['input_CT_pre'].cuda(non_blocking=True)
                    valid_input_CT_post = valid_data_dict['input_CT_post'].cuda(non_blocking=True)
                    valid_label_CT_pre = valid_data_dict['label_CT_pre'].cuda(non_blocking=True)
                    valid_label_CT_post = valid_data_dict['label_CT_post'].cuda(non_blocking=True)
                    bs, num_crops, c, h, w = valid_input_CT_pre.shape
                
                survival_time = valid_data_dict['survival_time'].cuda(non_blocking=True)
                event_indicator = valid_data_dict['event_indicator'].cuda(non_blocking=True)

            risk_score, survival_pred, _, _ = generator(
                [valid_input_CT_pre, valid_input_CT_post], 
                [valid_label_CT_pre, valid_label_CT_post],
            )

            five_year_survival_label = torch.tensor(
                [[1.0] if (st >= 60.0) else [0.0] for st in survival_time],
                device=risk_score.device
            )
            survival_loss = cox_criterion(risk_score, survival_time, event_indicator) + 0.1*ce_criterion(survival_pred, five_year_survival_label)
            
            valid_loss = survival_loss
            
            
            risk_scores_all.append(risk_score.detach().cpu())
            survival_times_all.append(survival_time.detach().cpu())
            events_all.append(event_indicator.detach().cpu())

            losses.update(valid_loss.item(), risk_score.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
    
    risk_scores_all = torch.cat(risk_scores_all).numpy()
    survival_times_all = torch.cat(survival_times_all).numpy()
    events_all = torch.cat(events_all).numpy()
    
    c_index = calculate_c_index(risk_scores_all, survival_times_all, events_all)
    
    writer.add_scalar('valid/loss', losses.avg, epoch)
    writer.add_scalar('valid/c_index', c_index, epoch)

    return losses.avg, c_index

def calculate_c_index(risk_scores, survival_times, events):
    """
    Calculate C-index (concordance index)
    Args:
        risk_scores: predicted risk scores
        survival_times: observed survival times
        events: event indicator (1 represents death, 0 represents censored)
    Returns:
        c_index: concordance index
    """
    from lifelines.utils import concordance_index
    return concordance_index(survival_times, -risk_scores, events)

def calculate_classification_metrics(y_pred, y_true):
    """
    Calculate classification evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro')
    }
    return metrics

def cox_ph_loss(risk, time, event):
    """Negative partial log-likelihood of Cox PH model"""
    order = torch.argsort(time, descending=False)
    risk, time, event = risk[order], time[order], event[order]
    log_cumsum_exp = torch.logcumsumexp(risk, dim=0)
    uncensored_likelihood = risk - log_cumsum_exp
    loss = -torch.sum(uncensored_likelihood * event)
    return loss / event.sum()  # normalize by event数

def cindex_surrogate_loss(risk_scores, times, events):
    """
    Args:
    - risk_scores: (N,) Tensor, model output risk scores, higher risk higher
    - times: (N,) Tensor, corresponding sample survival times
    - events: (N,) Tensor, 1 represents occurrence, 0 represents censored

    Returns:
    - surrogate C-index loss: the smaller the better
    """
    n = len(times)
    
    # Create a mask for pairwise comparisons
    # i: event sample, j: sample with later survival time
    idx_i = events == 1  # only consider event samples as anchor
    losses = []
    
    for i in range(n):
        if not idx_i[i]:
            continue
        for j in range(n):
            if times[i] < times[j]:
                diff = risk_scores[j] - risk_scores[i]
                loss_ij = torch.sigmoid(diff)
                losses.append(loss_ij)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=risk_scores.device)  # no comparable pairs
    return torch.stack(losses).mean()

def cleanup_nfs():
    max_retries = 3
    for i in range(max_retries):
        try:
            for f in os.listdir('.'):
                if f.startswith('.nfs'):
                    try:
                        os.remove(f)
                    except OSError:
                        continue
            break
        except OSError:
            if i == max_retries - 1:
                print("Warning: Could not remove some NFS files")
            time.sleep(1)

if __name__ == "__main__":
    main()

