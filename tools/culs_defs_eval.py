"""
Evaluation routines, metric calculations (AP, Precision, Recall, F-score),
and count statistics for OpenPCDet object detection models.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from easydict import EasyDict
import numpy as np
import pcdet.models
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import scipy.stats
import sklearn.metrics
import torch


def get_gt_preds_loss(
    cfg: EasyDict,
    model: torch.nn.Module,
    dataloader: Any
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Evaluates model and returns ground truths, predictions, and average loss.

    Args:
        cfg: Configuration dictionary.
        model: PyTorch detection model.
        dataloader: DataLoader containing evaluation batches.

    Returns:
        Tuple of (ground truths, prediction dictionaries, loss dict).
    """
    model.eval()
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.0
    ground_truths, predictions, loss = get_data_and_loss(model, dataloader)
    return ground_truths, predictions, loss


def evaluate_model_loss(
    cfg: EasyDict,
    model: torch.nn.Module,
    dataloader: Any,
    pr_curve: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Evaluates model and returns predictions alongside the loss metric.

    Args:
        cfg: Configuration dictionary.
        model: PyTorch detection model.
        dataloader: DataLoader containing evaluation batches.
        pr_curve: Unused flag kept for backward compatibility.

    Returns:
        Tuple of (prediction dictionaries, loss dict).
    """
    model.eval()
    _, predictions, loss = get_data_and_loss(model, dataloader)
    return predictions, {'loss': loss}


def evaluate_model_scores_per_sample(
    cfg: EasyDict,
    model: torch.nn.Module,
    dataloader: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Calculates detection metrics evaluated individually for each sample frame.

    Args:
        cfg: Configuration dictionary.
        model: PyTorch detection model.
        dataloader: DataLoader containing evaluation batches.

    Returns:
        Tuple of (prediction dictionaries, per-sample metrics dictionary).
    """
    metrics: Dict[str, Any] = {}
    model.eval()
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.0
    ground_truths, predictions = get_predictions_only(model, dataloader)
    scores = [x / 100.0 for x in range(1, 100, 1)]
    metrics.update(
        get_detection_metrics_per_sample(
            ground_truths, predictions, scores, cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
        )
    )
    return predictions, metrics


def evaluate_model_scores(
    cfg: EasyDict,
    model: torch.nn.Module,
    dataloader: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs full evaluation pipeline including count and detection metrics across folds.

    Args:
        cfg: Configuration dictionary.
        model: PyTorch detection model.
        dataloader: DataLoader containing evaluation batches.

    Returns:
        Tuple of (prediction dictionaries, comprehensive metrics dictionary).
    """
    tick_1 = time.perf_counter()
    metrics: Dict[str, Any] = {}
    model.eval()
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.0
    ground_truths, predictions = get_predictions_only(model, dataloader)

    scores = [x / 100.0 for x in range(1, 100, 1)]

    metrics.update(
        get_count_metrics(ground_truths, predictions, scores)
    )

    metrics.update(
        {
            "per_sample": get_detection_metrics_per_sample(
                ground_truths, predictions, scores, cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
            )
        }
    )
    metrics.update(
        get_detection_metrics(
            ground_truths, predictions, scores, cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
        )
    )
    tick_4 = time.perf_counter()
    metrics['eval_time'] = tick_4 - tick_1
    return predictions, metrics


def get_detection_metrics_per_sample(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    confidence_thresholds: List[float],
    iou_thresholds: List[float]
) -> Dict[str, Dict[str, float]]:
    """Computes AP and maximum F-score individually for each predicted frame.

    Args:
        ground_truths: List of ground-truth batches.
        predictions: List of sample prediction dictionaries.
        confidence_thresholds: List of confidence score cutoffs.
        iou_thresholds: List of 3D IoU threshold values.

    Returns:
        Dictionary mapping frame_id to its calculated metric scores.
    """
    result: Dict[str, Dict[str, float]] = {}
    cf = [np.array(x['score']) for x in predictions]
    iou = get_raw_iou(ground_truths, predictions, confidence_thresholds[0])
    iou = [np.array(x) for x in iou]

    for pred, iou_s, cf_s in zip(predictions, iou, cf):
        sample_result: Dict[str, float] = {}
        for iou_t in iou_thresholds:
            all_p: List[float] = []
            all_r: List[float] = []
            all_f: List[float] = []
            for cf_t in confidence_thresholds:
                iou_s_t, cf_s_t = drop_low_confidences_detection_in_sample(iou_s, cf_s, cf_t)
                tp, fp, fn = iou_to_truths(iou_s_t, iou_t)
                p, r, f = get_pr_and_score(tp, fp, fn)
                all_p.append(p)
                all_r.append(r)
                all_f.append(f)

            ap = 0.0
            cf_len = len(confidence_thresholds)
            for i in range(cf_len):
                ri = all_r[i]
                rj = all_r[i + 1] if (i != cf_len - 1) else 0.0
                pi = all_p[i]
                ap += (ri - rj) * pi

            sample_result[f"ap_{iou_t}"] = ap
            sample_result[f"fscore_{iou_t}"] = max(all_f) if all_f else 0.0
        result[pred["frame_id"]] = sample_result

    return result


def drop_low_confidences_detection_in_sample(
    iou: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters out detections below the given confidence score for a single sample.

    Args:
        iou: 2D array of IoU values (ground_truths x predictions).
        scores: 1D array of prediction confidence scores.
        threshold: Confidence threshold cutoff.

    Returns:
        Tuple of (filtered iou matrix, filtered scores array).
    """
    newlen = len([x for x in scores if x >= threshold])
    scores = scores[:newlen]
    iou = iou[:, :newlen]
    return iou, scores


def get_detection_metrics(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    confidence_thresholds: List[float],
    iou_thresholds: List[float]
) -> Dict[str, Any]:
    """Computes global dataset-level detection metrics (Precision, Recall, F-score, AP).

    Args:
        ground_truths: List of ground-truth batches.
        predictions: List of sample prediction dictionaries.
        confidence_thresholds: List of confidence score cutoffs.
        iou_thresholds: List of 3D IoU threshold values.

    Returns:
        Dictionary containing summary and per-confidence detection metrics.
    """
    result: Dict[str, Any] = {
        "detection_per_cf": {}
    }
    confidence_scores = [np.array(x['score']) for x in predictions]
    iou = get_raw_iou(ground_truths, predictions, confidence_thresholds[0])
    iou = [np.array(x) for x in iou]
    first = True

    for cf_t in confidence_thresholds:
        drop_low_confidences_detection(iou, confidence_scores, cf_t)
        score_result: Dict[str, float] = {}
        for iou_t in iou_thresholds:
            sum_tp, sum_fp, sum_fn = 0, 0, 0
            samples_p: List[float] = []
            samples_r: List[float] = []
            samples_f: List[float] = []
            for sample in iou:
                sample_tp, sample_fp, sample_fn = iou_to_truths(sample, iou_t)
                sample_p, sample_r, sample_f = get_pr_and_score(sample_tp, sample_fp, sample_fn)
                sum_tp += sample_tp
                sum_fp += sample_fp
                sum_fn += sample_fn
                samples_p.append(sample_p)
                samples_r.append(sample_r)
                samples_f.append(sample_f)

            sum_p, sum_r, sum_f = get_pr_and_score(sum_tp, sum_fp, sum_fn)
            score_result[f"precis_sum_{iou_t}"] = sum_p
            score_result[f"recall_sum_{iou_t}"] = sum_r
            score_result[f"fscore_sum_{iou_t}"] = sum_f

            avg_p, avg_r, avg_f = float(np.average(samples_p)), float(np.average(samples_r)), float(np.average(samples_f))
            score_result[f"precis_avg_{iou_t}"] = avg_p
            score_result[f"recall_avg_{iou_t}"] = avg_r
            score_result[f"fscore_avg_{iou_t}"] = avg_f

            if first:
                result[f"fscore_sum_{iou_t}"] = sum_f
                result[f"fscore_avg_{iou_t}"] = avg_f
                result[f"fscore_sum_{iou_t}_score"] = cf_t
                result[f"fscore_avg_{iou_t}_score"] = cf_t
            else:
                for key, var in zip(["fscore_sum", "fscore_avg"], [sum_f, avg_f]):
                    if var > result[f"{key}_{iou_t}"]:
                        result[f"{key}_{iou_t}"] = var
                        result[f"{key}_{iou_t}_score"] = cf_t
        first = False
        result['detection_per_cf'].update({cf_t: score_result})

    for iou_t in iou_thresholds:
        for typ in ['sum', 'avg']:
            cf_t = result[f"fscore_{typ}_{iou_t}_score"]
            for pr in ['precis', 'recall']:
                result[f"{pr}_{typ}_{iou_t}"] = result["detection_per_cf"][cf_t][f"{pr}_{typ}_{iou_t}"]

            ap = 0.0
            cf_len = len(confidence_thresholds)
            det_dict = result['detection_per_cf']
            for i in range(cf_len):
                ri = det_dict[confidence_thresholds[i]][f"recall_{typ}_{iou_t}"]
                rj = 0.0 if (i == cf_len - 1) else det_dict[confidence_thresholds[i + 1]][f"recall_{typ}_{iou_t}"]
                pi = det_dict[confidence_thresholds[i]][f"precis_{typ}_{iou_t}"]
                ap += (ri - rj) * pi
            result[f'ap_{typ}_{iou_t}'] = ap

    return result


def get_raw_iou(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    min_score: float = 0.0
) -> List[torch.Tensor]:
    """Calculates pairwise 3D IoU tensors for each sample using CUDA kernels.

    Args:
        ground_truths: List of ground-truth sample dictionaries.
        predictions: List of model prediction dictionaries.
        min_score: Unused score cutoff kept for backward compatibility.

    Returns:
        List of 2D IoU tensors on CPU.
    """
    result: List[torch.Tensor] = []
    for sample_gt, sample_pred in zip(ground_truths, predictions):
        boxes_gt = [np.asarray(x[:-1].to('cpu')) for x in sample_gt['gt_boxes'][0]]
        boxes_pred = sample_pred['boxes_lidar']
        boxes_gt_t = torch.tensor(boxes_gt, device='cuda', dtype=torch.float32)
        boxes_pred_t = torch.tensor(boxes_pred, device='cuda', dtype=torch.float32)
        iou = boxes_iou3d_gpu(boxes_gt_t, boxes_pred_t).to('cpu')
        result.append(iou)
    return result


def drop_low_confidences_detection(
    iou: List[np.ndarray],
    scores: List[np.ndarray],
    threshold: float
) -> None:
    """In-place filters predictions below threshold across all samples.

    Args:
        iou: List of IoU matrices modified in-place.
        scores: List of score arrays modified in-place.
        threshold: Confidence threshold cutoff.
    """
    for i in range(len(scores)):
        newlen = len([x for x in scores[i] if x >= threshold])
        scores[i] = scores[i][:newlen]
        iou[i] = iou[i][:, :newlen]


def get_count_metrics(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    confidence_thresholds: List[float]
) -> Dict[str, Any]:
    """Evaluates plant count regression performance (RMSE, MAE, R-squared).

    Args:
        ground_truths: List of ground-truth sample dictionaries.
        predictions: List of model prediction dictionaries.
        confidence_thresholds: List of confidence score cutoffs.

    Returns:
        Dictionary containing overall and per-threshold count statistics.
    """
    result: Dict[str, Any] = {'count_per_cf': {}}
    num_gt = np.array([len(x['gt_boxes'][0]) for x in ground_truths])
    confidence_scores = [np.array(x['score']) for x in predictions]

    b_rsq, b_rmse, b_mae = 0.0, math.inf, math.inf
    s_rsq, s_rmse, s_mae = 0.0, 0.0, 0.0

    for cf_t in confidence_thresholds:
        drop_low_confidences_count(confidence_scores, cf_t)
        num_ps = np.array([len(x) for x in confidence_scores])
        rsq = float(scipy_correlation(num_gt, num_ps) ** 2)
        rmse = float(sklearn.metrics.root_mean_squared_error(num_gt, num_ps))
        mae = float(sklearn.metrics.mean_absolute_error(num_gt, num_ps))

        score_result = {
            'rmse': rmse,
            'mae': mae,
            'rsq': rsq,
        }
        result['count_per_cf'].update({f"{cf_t}": score_result})

        if rmse < b_rmse:
            b_rmse = rmse
            s_rmse = cf_t
        if mae < b_mae:
            b_mae = mae
            s_mae = cf_t
        if rsq > b_rsq:
            b_rsq = rsq
            s_rsq = cf_t

    result['rmse'] = b_rmse
    result['mae'] = b_mae
    result['rsq'] = b_rsq
    result['rmse_score'] = s_rmse
    result['mae_score'] = s_mae
    result['rsq_score'] = s_rsq
    return result


def drop_low_confidences_count(scores: List[np.ndarray], threshold: float) -> None:
    """In-place filters score lists by threshold for count evaluation.

    Args:
        scores: List of score arrays modified in-place.
        threshold: Confidence cutoff threshold.
    """
    for i in range(len(scores)):
        newlen = len([x for x in scores[i] if x >= threshold])
        scores[i] = scores[i][:newlen]


def scipy_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Pearson correlation coefficient between two 1D arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Pearson correlation statistic, or 0.0 if undefined.
    """
    try:
        result = scipy.stats.pearsonr(a, b).statistic
    except ValueError:
        return 0.0
    if math.isnan(result):
        return 0.0
    return float(result)


def pairwise_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Fast vectorised Pearson correlation implementation.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Correlation coefficient value.
    """
    if a.shape != b.shape:
        return 0.0
    if np.any(a == b):
        return 1.0
    am = a - np.mean(a, axis=0, keepdims=True)
    bm = b - np.mean(b, axis=0, keepdims=True)
    return float(
        (
            am.T @ bm / (
                np.sqrt(np.sum(am ** 2, axis=0, keepdims=True)).T *
                np.sqrt(np.sum(bm ** 2, axis=0, keepdims=True))
            )
        )[0]
    )


def get_predictions_only(
    model: torch.nn.Module,
    dataloader: Any
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs model forward pass over dataloader without computing training loss.

    Args:
        model: PyTorch detection model.
        dataloader: DataLoader containing sample batches.

    Returns:
        Tuple of (ground truth batches, predicted bounding box dictionaries).
    """
    ground_truths: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for sample_gt in dataloader:
        sample_pred = make_prediction(model, sample_gt, dataloader.dataset)
        ground_truths.append(sample_gt)
        predictions.append(sample_pred[0])
    return ground_truths, predictions


def get_data_and_loss(
    model: torch.nn.Module,
    dataloader: Any
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Runs forward pass and extracts dense head RPN loss per sample.

    Args:
        model: Detection network with dense_head module.
        dataloader: DataLoader containing sample batches.

    Returns:
        Tuple of (ground truths, predictions, average loss dict).
    """
    ground_truths: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    losses = np.empty(len(dataloader), dtype=np.float64)

    for i, sample_gt in enumerate(dataloader):
        sample_pred = make_prediction(model, sample_gt, dataloader.dataset)
        ground_truths.append(sample_gt)
        predictions.append(sample_pred[0])
        losses[i] = model.dense_head.get_loss()[1]['rpn_loss']

    return ground_truths, predictions, {"loss": float(np.average(losses))}


def get_point_data_and_loss(
    model: torch.nn.Module,
    dataloader: Any
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Runs forward pass and extracts point and ROI head loss for PointRCNN.

    Args:
        model: Detection network with point_head and roi_head.
        dataloader: DataLoader containing sample batches.

    Returns:
        Tuple of (ground truths, predictions, loss breakdown dict).
    """
    ground_truths: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    losses = np.empty(len(dataloader), dtype=np.float64)
    losses_point = np.empty(len(dataloader), dtype=np.float64)
    losses_roi = np.empty(len(dataloader), dtype=np.float64)

    for i, sample_gt in enumerate(dataloader):
        sample_pred = make_prediction(model, sample_gt, dataloader.dataset)
        ground_truths.append(sample_gt)
        predictions.append(sample_pred[0])
        loss_point = model.point_head.get_loss()[0]
        loss_roi = model.roi_head.get_loss()[0]
        loss = loss_point + loss_roi
        losses[i] = loss
        losses_point[i] = loss_point
        losses_roi[i] = loss_roi

    loss_dict = {
        "loss": float(np.average(losses)),
        "loss_point": float(np.average(losses_point)),
        "loss_roi": float(np.average(losses_roi))
    }
    return ground_truths, predictions, loss_dict


def make_prediction(
    model: torch.nn.Module,
    sample_gt: Dict[str, Any],
    dataset: Any
) -> List[Dict[str, Any]]:
    """Transfers sample to GPU, performs inference, and generates bounding box dicts.

    Args:
        model: PyTorch detection network.
        sample_gt: Input ground truth sample dictionary.
        dataset: OpenPCDet dataset object providing generate_prediction_dicts.

    Returns:
        List of formatted prediction dictionaries.
    """
    pcdet.models.load_data_to_gpu(sample_gt)
    with torch.no_grad():
        pred_dicts, _ = model(sample_gt)
    return dataset.generate_prediction_dicts(
        sample_gt, pred_dicts, dataset.class_names, None
    )


def iou_to_truths(iou: np.ndarray, threshold: float) -> Tuple[int, int, int]:
    """Converts a pairwise IoU matrix into TP, FP, and FN counts using greedy matching.

    Args:
        iou: 2D array of IoU values between ground truths and predictions.
        threshold: IoU threshold for considering a detection as true positive.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives).
    """
    init_gts, init_preds = iou.shape
    if init_preds == 0:
        return 0, 0, init_gts

    tp = 0
    fn = 0
    for igt in range(init_gts):
        if iou.shape[1] == 0:
            fn += init_gts - igt
            break
        gt = iou[igt]
        imax = int(np.argmax(gt))
        if gt[imax] > threshold:
            tp += 1
            iou = np.delete(iou, imax, axis=1)
        else:
            fn += 1
    fp = iou.shape[1]
    return tp, fp, fn


def get_pr_and_score(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculates precision, recall, and harmonic F-score from confusion counts.

    Args:
        tp: Number of true positive detections.
        fp: Number of false positive detections.
        fn: Number of false negative detections.

    Returns:
        Tuple of (precision, recall, f_score).
    """
    if (tp + fp) == 0:
        return 1.0, 0.0, 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(p), float(r), float(f)