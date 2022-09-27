#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on huge images.

Example:
```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
python demo/huge_image_demo.py \
    demo/dota_demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_v3.py \
    checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
```
"""  # nowq

from argparse import ArgumentParser
import argparse
import os
import os.path as osp
import time
import numpy as np
from osgeo import gdal
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
from mmrotate.apis import inference_detector_by_patches
from mmdet.apis import init_detector, show_result_pyplot

def parse_args():
    parser = argparse.ArgumentParser(
        description='Patch Mode test (and eval) a model')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--gsd', action='store_true', help='gsd filtering')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--min-ship', type=float, default=0.0)
    parser.add_argument('--max-ship', type=float, default=0.0)
    parser.add_argument('--min-vehicle', type=float, default=0.0)
    parser.add_argument('--max-vehicle', type=float, default=0.0)
    parser.add_argument('--min-airplane', type=float, default=0.0)
    parser.add_argument('--max-airplane', type=float, default=0.0)
    args = parser.parse_args()
    return args

def main(args):

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)
    
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    # # build the model from a config file and a checkpoint file
    cfg.model.train_cfg = None                                       # lcg
    model = init_detector(args.config, args.checkpoint, device=args.device)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    outputs = []
    range_size = np.array([[args.min_ship, args.max_ship], [args.min_vehicle, args.max_vehicle], [args.min_airplane, args.max_airplane]])
    file_log = os.environ.get('INFER_DIRS') + '/list_' + timestamp + '.log'
    f_log = open(file_log, 'w')
    for i in range(len(dataset.img_ids)):
        args.img = './' + dataset.img_prefix + '/' + dataset.data_infos[i]['filename']
        f_log.write(f'{args.img}\n')
        args.out_file = os.environ.get('INFER_DIRS') + '/' + dataset.img_ids[i] + '_inf.tif'
        result = inference_detector_by_patches(model, args.img, args.patch_sizes,
                                            args.patch_steps, args.img_ratios,
                                            args.merge_iou_thr)
        str = ''
        gsd = 0.0
        raster = gdal.Open(args.img)
        try: 
            gt = raster.GetGeoTransform()
        except:
            gsd = -1.0
            f_log.write(f'ERROR 1: {args.img}\n')

        if gsd != -1.0:
            if gt[1] >= 0.01: gsd = gt[1]
            else: gsd = gt[1] * 100000
        if args.gsd is not False and gsd > 0.0:
            file_gsd = os.environ.get('INFER_DIRS') + '/GSD_' + dataset.img_ids[i] + '_' + timestamp + '.log'

            for ncls in range(len(result)):
                if len(result[ncls]) == 0: continue
                ndet = -1
                while True:
                    ndet += 1
                    try:
                        if result[ncls][ndet][2] > result[ncls][ndet][3]: length = result[ncls][ndet][2]
                        else: length = result[ncls][ndet][3]
                    except:
                        f_log.write(f'ERROR 2: {args.img}, {ncls}, {ndet}\n')
                        break
                    size = length * gsd
                    if range_size[ncls, :][1] < size or range_size[ncls, :][0] > size:
                        str += f'gsd: {gsd}, cls:{dataset.CLASSES[ncls]}, length: {length}, size: {length * gsd}, location: {result[ncls][ndet][0]}, {result[ncls][ndet][1]}, score: {result[ncls][ndet][5]} \n'
                        result[ncls] = np.delete(result[ncls], ndet, 0)
                        ndet -= 1
                    if len(result[ncls]) == 0: break
            if str != '':
                f = open(file_gsd, 'w')
                f.write(str)
                f.close()
        show_result_pyplot(model, args.img, result, palette=args.palette, score_thr=args.show_score_thr, out_file=args.out_file)
        outputs.append(result)
    f_log.close()

    args.out = os.environ.get('INFER_DIRS') + '/output_' + timestamp + '.pkl'

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)    

if __name__ == '__main__':
    args = parse_args()
    main(args)
