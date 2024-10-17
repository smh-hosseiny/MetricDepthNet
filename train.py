import argparse
import logging
import os
import pprint
import random
import warnings
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter    
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.syns import SYNSDataset
from dataset.nyu import NYU_Depth_V2
from unidepth.models import UniDepthV2
from util.dist_helper import setup_distributed
from util.loss import DepthLoss
from util.metric import eval_depth
from util.utils import init_log

def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--dataset', default="hypersim,vkitti,nyu,syns")
    parser.add_argument('--img-size', default=476, type=int)
    parser.add_argument('--min-depth', default=0.01, type=float)
    parser.add_argument('--max-depth', default=80, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--bs', default=6, type=int)
    parser.add_argument('--lr', default=0.000005, type=float)
    parser.add_argument('--pretrained-from', type=str)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    return parser.parse_args()

def setup_environment():
    warnings.simplefilter('ignore', np.RankWarning)
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    cudnn.enabled = True
    cudnn.benchmark = True

def create_logger(save_path, rank):
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    if rank == 0:
        writer = SummaryWriter(save_path)
    else:
        writer = None
    return logger, writer

def load_datasets(args, size):
    datasets = {
        'nyu': (NYU_Depth_V2, 'dataset/splits/nyu/train.txt', 'dataset/splits/nyu/val.txt'),
        'vkitti': (VKITTI2, 'dataset/splits/vkitti2/vkitti_train.txt', 'dataset/splits/vkitti2/vkitti_val.txt'),
        'syns': (SYNSDataset, 'dataset/splits/syns/syns_train.txt', 'dataset/splits/syns/syns_val.txt'),
        'hypersim': (Hypersim, 'dataset/splits/hypersim/partial_train.txt', None)
    }

    
    combined_trainset, combined_valset = [], []

    for dataset_name in args.dataset:
        if dataset_name in datasets:
            DatasetClass, train_path, val_path = datasets[dataset_name]
            if os.path.exists(train_path):
                combined_trainset.append(DatasetClass(train_path, 'train', size=size))
            if val_path and os.path.exists(val_path):
                combined_valset.append(DatasetClass(val_path, 'val', size=size))
            
             # Print the formatted output
            train_len = len(combined_trainset[-1]) if combined_trainset else 'N/A'
            val_len = len(combined_valset[-1]) if combined_valset else 'N/A'
            print(
                f"{dataset_name.ljust(10)} train set: {str(train_len).ljust(10)} val set: {str(val_len).ljust(10)}"
            )



    if not combined_trainset:
        warnings.warn(
            "No valid training datasets found. Please check dataset paths and arguments.",
            UserWarning
        )

    if not combined_valset:
        warnings.warn(
            "No valid validation datasets found. Please check dataset paths and arguments.",
            UserWarning
        )

    return ConcatDataset(combined_trainset), ConcatDataset(combined_valset)



def load_model(args, config, local_rank):
    model = UniDepthV2(config)
    # model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    if args.pretrained_from:
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu')['model'], strict=True)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    return model

def main():
    args = parse_args()
    args.dataset = args.dataset.split(',')

    setup_environment()

    rank, world_size = setup_distributed(port=args.port)
    logger, writer = create_logger(args.save_path, rank)

    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    size = (462, 616)
    combined_trainset, combined_valset = load_datasets(args, size)
    
    trainloader = DataLoader(combined_trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True,
                             sampler=DistributedSampler(combined_trainset, shuffle=True))
    valloader = DataLoader(combined_valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True,
                           sampler=DistributedSampler(combined_valset, shuffle=True))
    
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    with open('configs/config_v2_vits14.json') as f:
        config = json.load(f)
    
    model = load_model(args, config, local_rank)

    criterion = DepthLoss(is_metric=True, include_abs_rel=True).cuda(local_rank)
    
    optimizer = AdamW([
        {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
        {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)
    previous_best = {metric: 100 if metric not in ['d1', 'd2', 'd3'] else 0 for metric in ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']}

    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, Metrics: {}'.format(epoch, args.epochs, previous_best))
        
        trainloader.sampler.set_epoch(epoch + 1)
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            if random.random() < 0.5:
                img, depth, valid_mask = img.flip(-1), depth.flip(-1), valid_mask.flip(-1)
            
            output = model(sample)
            pred = output["depth"]
            loss = criterion(pred.unsqueeze(1), depth.unsqueeze(1), ((valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)).unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
                if i % 100 == 0:
                    logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), lr, loss.item()))
        
        # Evaluation loop - modularized and reduced redundant code.
        model.eval()
        results, nsamples = evaluate_model(model, valloader, criterion, args, local_rank)
        
        if rank == 0:
            log_and_save_results(logger, writer, results, nsamples, epoch, previous_best, model, optimizer, args.save_path, config)

def evaluate_model(model, valloader, criterion, args, local_rank):
    results = {metric: torch.tensor([0.0]).cuda() for metric in ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']}
    nsamples = torch.tensor([0.0]).cuda()
    
    for sample in valloader:
        img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
        with torch.no_grad():
            pred = model(sample)["depth"].squeeze()
        
        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        if valid_mask.sum() < 10:
            continue
        
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    torch.distributed.barrier()
    for k in results.keys():
        dist.reduce(results[k], dst=0)
    dist.reduce(nsamples, dst=0)
    return results, nsamples

def log_and_save_results(logger, writer, results, nsamples, epoch, previous_best, model, optimizer, save_path, config):
    logger.info('==========================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*[(v / nsamples).item() for v in results.values()]))
    logger.info('==========================================================================================')
    for name, metric in results.items():
        writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
    
    for k in results.keys():
        if k in ['d1', 'd2', 'd3']:
            previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        else:
            previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'previous_best': previous_best,
        'config': config
    }
    torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))

if __name__ == '__main__':
    main()