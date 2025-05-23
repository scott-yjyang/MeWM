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

# from torcheval.metrics.aggregation.auc import AUC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import shutil
import warnings
import builtins
import random
import math
import sys
from shutil import copyfile
import csv

from dataset import ImageDataset, BalancedDistributedSampler
from model.utils import get_model
from utils import AverageMeter, calculate_accuracy, save_checkpoint, adjust_learning_rate, ProgressMeter

from config import create_arg_parser
from survival_loss import NLLSurvivalLoss

def main():
    args = create_arg_parser()
    
    if args.gpu is not None:
        args.gpu = [int(i) for i in args.gpu.split(',')]
    print(args.gpu)

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
                args=(args.ngpus_per_node, args, save_dir))
    else:
        main_worker(args.gpu, 1, args, save_dir)

def main_worker(gpu, ngpus_per_node, args, save_dir):
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

    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    print("Epoch: {}".format(epoch))

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

    # criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    bce_criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
    cox_criterion = NLLSurvivalLoss().cuda(args.gpu)


    valid_dataset = ImageDataset(args, mode='valid')
    dataloader_valid = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,                   num_workers=args.num_workers, pin_memory=True)
    
    _, c_index = valid(dataloader_valid, generator, bce_criterion, cox_criterion, args, writer)
    print("c_index: ", c_index)



def valid(dataloader_valid, generator, bce_criterion, cox_criterion, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(dataloader_valid),
        [batch_time, data_time, losses],
        prefix="Valid",
    )
    
    risk_scores_all = []
    survival_times_all = []
    events_all = []
    patient_ids_all = []
    ce_criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
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
                    valid_input_CT_pre = valid_input_CT_pre.squeeze(0).unsqueeze(1)
                    valid_input_CT_post = valid_input_CT_post.squeeze(0).unsqueeze(1)
                    valid_label_CT_pre = valid_label_CT_pre.squeeze(0).unsqueeze(1)
                    valid_label_CT_post = valid_label_CT_post.squeeze(0).unsqueeze(1)
                
                survival_time = valid_data_dict['survival_time'].cuda(non_blocking=True)
                event_indicator = valid_data_dict['event_indicator'].cuda(non_blocking=True)
                event_indicator = event_indicator.squeeze(0).unsqueeze(1)
                survival_time = survival_time.squeeze(0).unsqueeze(1)

            risk_score, survival_pred, _, _ = generator(
                [valid_input_CT_pre, valid_input_CT_post], 
                [valid_label_CT_pre, valid_label_CT_post]
            )
            valid_loss = cox_criterion(risk_score, survival_time, event_indicator) + ce_criterion(survival_pred, event_indicator)

            risk_scores_all.append(torch.mean(risk_score, dim=0).detach().cpu())
            survival_times_all.append(torch.mean(survival_time, dim=0).detach().cpu())
            events_all.append(torch.mean(event_indicator, dim=0).detach().cpu())
            patient_ids_all.append(valid_data_dict['patient_id'][0])

            losses.update(valid_loss.item(), risk_score.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
    
    risk_scores_all = torch.cat(risk_scores_all).numpy()
    survival_times_all = torch.cat(survival_times_all).numpy()
    events_all = torch.cat(events_all).numpy()
    c_index = calculate_c_index(risk_scores_all, survival_times_all, events_all)
    # writer.add_scalar('valid/loss', losses.avg)
    # writer.add_scalar('valid/c_index', c_index)
    


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

