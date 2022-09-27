# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
import pandas as pd
import copy
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from terminaltables import AsciiTable

    
def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp

    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_metric(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        #yms
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        f1_scores = 2 * (recalls * precisions) / (recalls + precisions)

        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'tp': tp,
            'fp': fp,
            'recall': recalls,
            'precision': precisions,
            'f1_score': f1_scores,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0

    mean_recall, mean_precision, mean_f1, mean_ap,\
    ship_recall, vehicle_recall, airplane_recall,\
    ship_precision, vehicle_precision, airplane_precision,\
    ship_f1, vehicle_f1, airplane_f1, \
    ship_ap, vehicle_ap, airplane_ap, \
        = print_metric_summary(mean_ap, eval_results, dataset, area_ranges, logger=logger)
        

    return mean_recall, mean_precision, mean_f1, mean_ap,\
        ship_recall, vehicle_recall, airplane_recall,\
        ship_precision, vehicle_precision, airplane_precision,\
        ship_f1, vehicle_f1, airplane_f1, \
        ship_ap, vehicle_ap, airplane_ap, eval_results

def print_metric_summary(mean_ap,
                         results,
                         dataset=None,
                         scale_ranges=None,
                         logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)
    
    # yms
    tps = np.zeros((num_scales, num_classes), dtype=np.float32)
    fps = np.zeros((num_scales, num_classes), dtype=np.float32)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    f1_scores = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            tps[:, i] = np.array(cls_result['tp'], ndmin=2)[:, -1]
            fps[:, i] = np.array(cls_result['fp'], ndmin=2)[:, -1]
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]
            f1_scores[:, i] = np.array(cls_result['f1_score'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']
    
    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
        
    header = ['class', 'gts', 'dets', 'tp', 'fp', 'fn', 'recall', 'precision', 'f1-score', 'ap'] 

    #yms
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            # if f1_scores[i, j] == np.nan:  # error
            # if np.isnan(f1_scores[i, j]):
            #     f1_scores[i, j] = 0.000
            row_data = [label_names[j], num_gts[i, j], results[j]['num_dets'],
                int(tps[i, j]),  int(fps[i, j]), num_gts[i, j] - int(tps[i, j]),
                f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}',
                f'{f1_scores[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
    
        if recalls[:,0][0].tolist() != 0 and recalls[:,1][0].tolist() != 0 and recalls[:,2][0].tolist() != 0: 
            mean_recall = round((recalls[:,0][0].tolist() + recalls[:,1][0].tolist() + recalls[:,2][0].tolist()) / 3, 4)
        elif recalls[:,0][0].tolist() != 0 and recalls[:,1][0].tolist() != 0 and recalls[:,2][0].tolist() == 0:     
            mean_recall = round((recalls[:,0][0].tolist() + recalls[:,1][0].tolist()) / 2, 4)      
        elif recalls[:,0][0].tolist() != 0 and recalls[:,1][0].tolist() == 0 and recalls[:,2][0].tolist() != 0:     
            mean_recall = round((recalls[:,0][0].tolist() + recalls[:,2][0].tolist()) / 2, 4)     
        elif recalls[:,0][0].tolist() == 0 and recalls[:,1][0].tolist() != 0 and recalls[:,2][0].tolist() != 0:     
            mean_recall = round((recalls[:,1][0].tolist() + recalls[:,2][0].tolist()) / 2, 4)   
        elif recalls[:,0][0].tolist() != 0 and recalls[:,1][0].tolist() == 0 and recalls[:,2][0].tolist() == 0:     
            mean_recall = round(recalls[:,0][0].tolist(), 4)
        elif recalls[:,0][0].tolist() == 0 and recalls[:,1][0].tolist() != 0 and recalls[:,2][0].tolist() == 0:     
            mean_recall = round(recalls[:,1][0].tolist(), 4)    
        elif recalls[:,0][0].tolist() == 0 and recalls[:,1][0].tolist() == 0 and recalls[:,2][0].tolist() != 0:     
            mean_recall = round(recalls[:,2][0].tolist(), 4)   
        else:     
            mean_recall = 0.0
        
        if precisions[:,0][0].tolist() != 0 and precisions[:,1][0].tolist() != 0 and precisions[:,2][0].tolist() != 0: 
            mean_precision = round((precisions[:,0][0].tolist() + precisions[:,1][0].tolist() + precisions[:,2][0].tolist()) / 3, 4)
        elif precisions[:,0][0].tolist() != 0 and precisions[:,1][0].tolist() != 0 and precisions[:,2][0].tolist() == 0:     
            mean_precision = round((precisions[:,0][0].tolist() + precisions[:,1][0].tolist()) / 2, 4)      
        elif precisions[:,0][0].tolist() != 0 and precisions[:,1][0].tolist() == 0 and precisions[:,2][0].tolist() != 0:     
            mean_precision = round((precisions[:,0][0].tolist() + precisions[:,2][0].tolist()) / 2, 4)     
        elif precisions[:,0][0].tolist() == 0 and precisions[:,1][0].tolist() != 0 and precisions[:,2][0].tolist() != 0:     
            mean_precision = round((precisions[:,1][0].tolist() + precisions[:,2][0].tolist()) / 2, 4)   
        elif precisions[:,0][0].tolist() != 0 and precisions[:,1][0].tolist() == 0 and precisions[:,2][0].tolist() == 0:     
            mean_precision = round(precisions[:,0][0].tolist(), 4)
        elif precisions[:,0][0].tolist() == 0 and precisions[:,1][0].tolist() != 0 and precisions[:,2][0].tolist() == 0:     
            mean_precision = round(precisions[:,1][0].tolist(), 4)    
        elif precisions[:,0][0].tolist() == 0 and precisions[:,1][0].tolist() == 0 and precisions[:,2][0].tolist() != 0:     
            mean_precision = round(precisions[:,2][0].tolist(), 4)    
        else:     
            mean_precision = 0.0
                                                                                        
        # if f1_scores[:,0][0].tolist() != 0 and f1_scores[:,1][0].tolist() != 0 and f1_scores[:,2][0].tolist() != 0:
        #     mean_f1 = round((f1_scores[:,0][0].tolist() + f1_scores[:,1][0].tolist() + f1_scores[:,2][0].tolist()) / 3, 4)
        # elif f1_scores[:,0][0].tolist() != 0 and f1_scores[:,1][0].tolist() != 0 and f1_scores[:,2][0].tolist() == 0:     
        #     mean_f1 = round((f1_scores[:,0][0].tolist() + f1_scores[:,1][0].tolist()) / 2, 4)      
        # elif f1_scores[:,0][0].tolist() != 0 and f1_scores[:,1][0].tolist() == 0 and f1_scores[:,2][0].tolist() != 0:     
        #     mean_f1 = round((f1_scores[:,0][0].tolist() + f1_scores[:,2][0].tolist()) / 2, 4)     
        # elif f1_scores[:,0][0].tolist() == 0 and f1_scores[:,1][0].tolist() != 0 and f1_scores[:,2][0].tolist() != 0:     
        #     mean_f1 = round((f1_scores[:,1][0].tolist() + f1_scores[:,2][0].tolist()) / 2, 4)   
        # elif f1_scores[:,0][0].tolist() != 0 and f1_scores[:,1][0].tolist() == 0 and f1_scores[:,2][0].tolist() == 0:     
        #     mean_f1 = round(f1_scores[:,0][0].tolist(), 4)
        # elif f1_scores[:,0][0].tolist() == 0 and f1_scores[:,1][0].tolist() != 0 and f1_scores[:,2][0].tolist() == 0:     
        #     mean_f1 = round(f1_scores[:,1][0].tolist(), 4)    
        # elif f1_scores[:,0][0].tolist() == 0 and f1_scores[:,1][0].tolist() == 0 and f1_scores[:,2][0].tolist() != 0:     
        #     mean_f1 = round(f1_scores[:,2][0].tolist(), 4)    
        # else:     
        #     mean_f1 = 0.0

        # lcg
        counter = 0
        sum = 0
        for nc in range(num_classes):
            if ~np.isnan(f1_scores[0, nc]) and num_gts[0,nc] != 0:
                counter += 1
                sum += f1_scores[0, nc]
        if counter == 0:
            mean_f1 = 0.0
        else:
            mean_f1 = sum / counter
                    
        mean_ap = round(mean_ap[i], 4)
        total_num_gts = num_gts[i, 0] + num_gts[i, 1] + num_gts[i, 2]
        total_num_dets = results[0]['num_dets'] + results[1]['num_dets'] + results[2]['num_dets']
        total_tps = int(tps[i, 0]) + int(tps[i, 1]) + int(tps[i, 2])
        total_fps = int(fps[i, 0]) + int(fps[i, 1]) + int(fps[i, 2])
        total_fns = total_num_gts - total_tps
        
        table_data.append(['total', total_num_gts,  total_num_dets, total_tps, total_fps, total_fns]) 
        table_data.append(['mean recall', '', '', '', '', '', f'{mean_recall:.3f}']) 
        table_data.append(['mean precision', '', '', '', '', '', '', f'{mean_precision:.3f}']) 
        table_data.append(['mean f1-score', '', '', '', '', '', '', '', f'{mean_f1:.3f}'])       
        table_data.append(['mAP', '', '', '', '', '', '', '', '', f'{mean_ap:.3f}'])   
        table = AsciiTable(table_data)     

        # lcg
        rows_before_total = 6
        border_line = copy.deepcopy(table.table[0:(table.table_width + 1)])
        length_rows_before_total = (table.table_width + 1) * (rows_before_total)
        new_table = copy.deepcopy(table.table[0:length_rows_before_total] + border_line + \
            table.table[length_rows_before_total:length_rows_before_total+(table.table_width + 1)] + border_line + \
            table.table[length_rows_before_total+(table.table_width + 1):])
        # print_log('\n' + table.table, logger=logger)          
        print_log('\n' + new_table, logger=logger)          
        print("\n")

        # lcg
        import os, time
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        output_folder_metric = ''
        output_file_metric = 'metric_' + timestamp
        try:
            if os.environ.get('CFG_MODE') == 'TRAIN_MODE':
                output_folder_metric = os.environ.get('TRAIN_DIRS')
                print('MODE: TRAIN_MODE')
            elif os.environ.get('CFG_MODE') == 'INFER_MODE':
                output_folder_metric = os.environ.get('INFER_DIRS')
                print('MODE: INFER_MODE')
        except:
            pass
        if output_folder_metric != ' ':
            file_metric = f'{output_folder_metric}/{output_file_metric}'
            f = open(file_metric, 'a') 
            f.write('\n')
            f.write(new_table)
            f.close()
            
        #yms
        return mean_recall, mean_precision, mean_f1, mean_ap,\
    round(recalls[:,0][0].tolist(), 4), round(recalls[:,1][0].tolist(), 4), round(recalls[:,2][0].tolist(), 4), \
    round(precisions[:,0][0].tolist(), 4), round(precisions[:,1][0].tolist(), 4), round(precisions[:,2][0].tolist(), 4), \
    round(f1_scores[:,0][0].tolist(), 4), round(f1_scores[:,1][0].tolist(), 4), round(f1_scores[:,2][0].tolist(), 4), \
    round(aps[:,0][0].tolist(), 4), round(aps[:,1][0].tolist(), 4), round(aps[:,2][0].tolist(), 4)
    
    
    
