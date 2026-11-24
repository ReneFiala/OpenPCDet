#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training and evaluation CLI script for OpenPCDet models supporting
early stopping, metrics tracking, and k-fold cross-validation runs.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numba
import scipy.stats
import torch

import culs_defs_callbacks as callbacks
import culs_defs_eval as culs_eval
import culs_defs_loading as culs_loading
import culs_defs_train as culs_training


def main(
    cfg_path: str,
    data_path: Optional[str],
    epochs: Optional[int],
    out_path: str,
    es_patience: int,
    es_warmup: int,
    es_obj: List[str],
    batch_size: int,
    augs: List[str],
    aug_val: bool,
    epochs_per_eval: int,
    refit: bool
) -> Dict[str, Any]:
    """Executes model training with custom CULS callbacks and evaluation pipeline.

    Args:
        cfg_path: Path to model YAML configuration file.
        data_path: Optional override for dataset root directory.
        epochs: Optional epoch count override.
        out_path: Directory path for exporting logs and checkpoints.
        es_patience: Early stopping patience epochs (0 disables early stopping).
        es_warmup: Warm-up epochs before early stopping takes effect.
        es_obj: Pair of strings [metric_name, 'min'|'max'].
        batch_size: Batch size per iteration.
        augs: List of data augmentation suffixes.
        aug_val: Whether to apply augmentations on validation split.
        epochs_per_eval: Run evaluation every n-th epoch.
        refit: If True, merges train and val splits and skips early stopping.

    Returns:
        Dictionary containing checkpoint and collected metric records.
    """
    culs_loading.ensure_determinism()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
    warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)

    cfg = culs_loading.load_config(cfg_path, epochs)

    if data_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = data_path

    if len(augs) > 0 and augs[0] != "__NONE":
        cfg.DATA_CONFIG.CULS_AUGMENTS = augs
        cfg.DATA_CONFIG.CULS_AUGMENTS_VAL = aug_val

    logger = culs_loading.create_console_logger()

    if not refit:
        train_data = culs_loading.create_loaders(cfg, logger, "train", epochs=epochs, batch_size=batch_size)
        val_data = culs_loading.create_loaders(cfg, logger, "val", epochs=epochs)
    else:
        train_data = culs_loading.create_loaders(cfg, logger, ["train", "val"], epochs=epochs, batch_size=batch_size)
        val_data = None

    test_data = culs_loading.create_loaders(cfg, logger, "test", epochs=epochs)

    model, optimizer, lr_sched, lr_wu_sched = culs_loading.build_model(cfg, train_data)

    epoch_callbacks: List[callbacks.CulsCallback] = []
    end_callbacks: List[callbacks.CulsCallback] = []

    if es_patience > 0 and not refit:
        epoch_callbacks.append(
            callbacks.EarlyStopCallback(epochs, es_obj[0], es_obj[1], es_patience, es_warmup, out_path)
        )

    epoch_callbacks.append(callbacks.CsvMetricsCallback(out_path, append=False))
    epoch_callbacks.append(
        callbacks.SaveBestCheckpointsCallback([("ap_sum_0.3", "max")], out_path)
    )

    end_callbacks.append(callbacks.CsvMetricsCallback(out_path, append=False))

    pickle_cb = callbacks.PickleMetricsCallback(out_path)
    epoch_callbacks.append(pickle_cb)
    end_callbacks.append(pickle_cb)

    val_eval_func = culs_eval.evaluate_model_scores if not refit else None

    result = culs_training.train_model(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_normal_scheduler=lr_sched,
        lr_warmup_scheduler=lr_wu_sched,
        logger=logger,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_eval_func=culs_eval.evaluate_model_scores,
        val_eval_func=val_eval_func,
        test_eval_func=culs_eval.evaluate_model_scores,
        epoch_callbacks=epoch_callbacks,
        train_callbacks=end_callbacks,
        epochs_per_eval=epochs_per_eval
    )
    return result


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments preserving compatibility with bash pipelines."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate OpenPCDet models with custom CULS callbacks."
    )
    parser.add_argument("config", type=str, help="Path to the model YAML config file.")
    parser.add_argument("output", type=str, help="Output folder path.")
    parser.add_argument(
        "--epochs", "-e", type=int, default=None, nargs="?",
        help="Number of epochs to train for. If None, use epochs from the config."
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default=None, metavar="PATH",
        help="Alternative dataset path. Does not affect the rest of the dataset configuration."
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, metavar="ID",
        help="ID of the GPU to use."
    )
    parser.add_argument(
        "--es-patience", type=int, default=0, metavar="EPOCHS",
        help="Patience in epochs for early stopping. If 0, early stopping is disabled (default: %(default)s)."
    )
    parser.add_argument(
        "--es-warmup", type=int, default=0, metavar="EPOCHS",
        help="Warm-up period in epochs for early stopping (default: %(default)s)."
    )
    parser.add_argument(
        "--es-objective", type=str, default=["loss", "min"], nargs="*",
        help="Two strings: target metric name and optimization goal ('min' or 'max')."
    )
    parser.add_argument("--batch-size", type=int, default=1, metavar="BZ", help="Batch size per step.")
    parser.add_argument(
        "--augs", type=str, nargs="*", default="",
        help="List of augmentations to use. Override of config's DATA_CONFIG.CULS_AUGMENTS."
    )
    parser.add_argument(
        "--aug-val", action="store_true",
        help="If set, apply augmentations with the validation set as well."
    )
    parser.add_argument(
        "--epochs-per-eval", type=int, default=1, metavar="EPOCHS",
        help="Only run evaluations every n-th epoch."
    )
    parser.add_argument(
        "--refit", "-r", action="store_true",
        help="Combine train and val sets together into one train set."
    )

    args = parser.parse_args()

    if len(args.es_objective) != 2:
        raise ValueError("--es-objective must have exactly two values specified (e.g. --es-objective ap_sum_0.3 max).")
    if args.es_objective[1] not in ["min", "max"]:
        raise ValueError("--es-objective 2nd value must be 'min' or 'max'.")
    if len(args.augs) == 1 and "@" in args.augs[0]:
        args.augs = args.augs[0].split("@")

    return args


def setup_cuda_device(gpu: int) -> None:
    """Sets active CUDA device safely with rollover fallback."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return

    device_count = torch.cuda.device_count()
    if gpu >= 0:
        gpu_id = gpu % device_count
        torch.cuda.set_device(gpu_id)
        if gpu >= device_count:
            print(f"GPU ID {gpu} out of range ({device_count}); rolling over to {gpu_id}.")
    else:
        print("Invalid GPU ID. Using 0.")
        torch.cuda.set_device(0)


if __name__ == "__main__":
    args = parse_args()
    setup_cuda_device(args.gpu)
    main(
        cfg_path=args.config,
        data_path=args.dataset,
        epochs=args.epochs,
        out_path=args.output,
        es_patience=args.es_patience,
        es_warmup=args.es_warmup,
        es_obj=args.es_objective,
        batch_size=args.batch_size,
        augs=args.augs,
        aug_val=args.aug_val,
        epochs_per_eval=args.epochs_per_eval,
        refit=args.refit
    )