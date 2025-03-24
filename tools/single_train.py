#!/usr/bin/env python
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Zhiqi Li
# ---------------------------------------------

from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmcv.utils import TORCH_VERSION, digit_version

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='skip validation during training')
    
    # GPU参数组
    group_gpus = parser.add_argument_group('GPU Options')
    group_gpus.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='specify GPU ids')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='enable deterministic mode')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='auto-scale learning rate')
    
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError('Cannot use both --options and --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated, use --cfg-options instead')
        args.cfg_options = args.options

    return args

def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 处理自定义导入
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)

    # 插件处理
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        # 修复路径处理：去除末尾的斜杠避免导入错误
        # Fix path handling: remove trailing slash to avoid import errors
        plugin_path = (cfg.plugin_dir if hasattr(cfg, 'plugin_dir') else osp.dirname(args.config)).rstrip('/')
        module_path = plugin_path.replace('/', '.')
        importlib.import_module(module_path)
        from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model

    # 设置cudnn
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 工作目录设置
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from

    # 强制单GPU模式
    if args.launcher == 'none':
        distributed = False
        if args.gpus > 1:
            warnings.warn('Forcing single-GPU mode')
            args.gpus = 1
        if args.gpu_ids and len(args.gpu_ids) > 1:
            warnings.warn('Using first GPU only')
            args.gpu_ids = [args.gpu_ids[0]]
    else:
        raise NotImplementedError("Distributed training disabled")

    # GPU配置
    cfg.gpu_ids = args.gpu_ids if args.gpu_ids else [0]
    torch.cuda.set_device(cfg.gpu_ids[0])

    # 学习率调整
    if args.autoscale_lr:
        cfg.optimizer.lr = cfg.optimizer.lr * args.gpus / 8

    # 初始化日志
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger_name = 'mmseg' if cfg.model.type == 'EncoderDecoder3D' else 'mmdet'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # 记录环境信息
    meta = {
        'env_info': collect_env(),
        'config': cfg.pretty_text,
        'seed': args.seed,
        'exp_name': osp.basename(args.config)
    }
    logger.info(f'Environment Info:\n{meta["env_info"]}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 设置随机种子
    if args.seed is not None:
        logger.info(f'Set seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # 构建数据集（关键修复点）
    train_dataset = build_dataset(cfg.data.train)
    datasets = [train_dataset]
    
    # 验证集处理
    if len(cfg.workflow) == 2:
        val_cfg = copy.deepcopy(cfg.data.val)
        val_cfg.pipeline = cfg.data.train.dataset.pipeline if 'dataset' in cfg.data.train else cfg.data.train.pipeline
        val_cfg.test_mode = False
        datasets.append(build_dataset(val_cfg))
    
    logger.info(f'Built {len(datasets)} dataset(s)')

    # 检查点配置（必须在数据集之后）
    if cfg.checkpoint_config:
        assert len(datasets) > 0, "Datasets not initialized!"
        cfg.checkpoint_config.meta = {
            'mmdet_version': mmdet_version,
            'mmseg_version': mmseg_version,
            'mmdet3d_version': mmdet3d_version,
            'config': cfg.pretty_text,
            'seed': cfg.seed,
            'CLASSES': datasets[0].CLASSES,
            'PALETTE': datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None
        }

    # 构建模型
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # 单GPU包装
    if not distributed and len(cfg.gpu_ids) > 0:
        model = model.cuda()
        if len(cfg.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=cfg.gpu_ids)
    
    logger.info(f'Model Architecture:\n{model}')

    # 调整数据加载参数
    if not distributed:
        cfg.data.workers_per_gpu = min(cfg.data.workers_per_gpu, 4)
        if args.gpus:
            cfg.data.samples_per_gpu = cfg.data.samples_per_gpu // args.gpus

    # 启动训练
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta=meta
    )

if __name__ == '__main__':
    # WSL2兼容性设置
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    main()
