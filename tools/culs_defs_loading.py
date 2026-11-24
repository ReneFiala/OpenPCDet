"""
Dataset loading, determinism, network initialization, and logging utilities
for OpenPCDet training and evaluation pipeline.
"""

from collections import namedtuple
import logging
from pathlib import Path
import random
from typing import Any, List, Optional, Tuple, Union

from easydict import EasyDict
import numpy as np
import pcdet
import pcdet.config
import pcdet.datasets
import pcdet.models
import torch
import train_utils.optimization as opt_utils

Dataset = namedtuple('Dataset', ['set', 'loader', 'sampler'])


def ensure_determinism(seed: int = 12345) -> None:
    """Sets random seeds and enables deterministic algorithms for reproducibility.

    Args:
        seed: Random seed value for PyTorch, Python random, and NumPy.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.use_deterministic_algorithms(True)


def load_trial_params(study: Any, cfg: EasyDict) -> None:
    """Loads hyperparameter tuning parameters from an Optuna study into configuration.

    Args:
        study: Optuna Study object containing the best trial.
        cfg: OpenPCDet EasyDict configuration to update in-place.
    """
    params = study.best_trial.params
    layer_nums: List[int] = []
    strides: List[int] = []
    num_filters: List[int] = []
    num_upsample_filters: List[int] = []

    for i in range(params["2d_length"]):
        layer_nums.append(params[f"2d_layer_num{i}"])
        num_filters.append(params[f"2d_filter{i}"])
        num_upsample_filters.append(params[f"2d_upsample_filter{i}"])
        strides.append(1)
    strides[-1] = params["2d_last_stride"]

    cfg.MODEL.BACKBONE_2D.LAYER_NUMS = layer_nums
    cfg.MODEL.BACKBONE_2D.NUM_FILTERS = num_filters
    cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS = num_upsample_filters
    cfg.MODEL.BACKBONE_2D.LAYER_STRIDES = strides
    cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES = strides

    anchors = []
    for i in range(params["dense_anchor_len"]):
        anchors.append([
            2 ** params[f"dense_anchor{i}_x"],
            2 ** params[f"dense_anchor{i}_y"],
            2 ** params[f"dense_anchor{i}_z"],
        ])
    cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG[0].anchor_sizes = anchors

    cfg.OPTIMIZATION.LR = params["opt_lr"]
    cfg.OPTIMIZATION.WEIGHT_DECAY = params["opt_weight_decay"]
    cfg.OPTIMIZATION.MOMENTUM = params["opt_momentum"]
    # must be 2 values here!
    cfg.OPTIMIZATION.MOMS = [
        params["opt_moms_0"],
        params["opt_moms_1"]
    ]
    cfg.OPTIMIZATION.PCT_START = params["opt_pct_start"]
    cfg.OPTIMIZATION.DIV_FACTOR = params["opt_div_factor"]
    cfg.OPTIMIZATION.LR_DECAY = params["opt_lr_decay"]
    cfg.OPTIMIZATION.WARMUP_EPOCH = params["opt_warmup"]
    cfg.OPTIMIZATION.LR_WARMUP = not (cfg.OPTIMIZATION.WARMUP_EPOCH == 0)
    cfg.OPTIMIZATION.GRAD_NORM_CLIP = params["opt_grad_norm_clip"]


def load_config(cfg_file: Union[str, Path], epochs: Optional[int] = None) -> EasyDict:
    """Loads OpenPCDet configuration from a YAML file.

    Args:
        cfg_file: Path to the YAML configuration file.
        epochs: Optional epoch count to overwrite cfg.OPTIMIZATION.NUM_EPOCHS.

    Returns:
        Populated EasyDict configuration object.
    """
    cfg_file_str = str(cfg_file)
    cfg = pcdet.config.cfg
    pcdet.config.cfg_from_yaml_file(cfg_file_str, cfg)
    cfg.TAG = Path(cfg_file_str).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file_str.split('/')[1:-1])
    if epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = epochs
    return cfg


def create_loaders(
    cfg: EasyDict,
    logger: logging.Logger,
    split_name: str,
    batch_size: int = 1,
    workers: int = 1,
    epochs: int = 0,
    no_shuffle: bool = False
) -> Dataset:
    """Builds OpenPCDet dataset, data loader, and sampler for the specified split.

    Args:
        cfg: Configuration dictionary.
        logger: Logger instance.
        split_name: Split name ('train', 'val', or 'test').
        batch_size: Batch size per iteration.
        workers: Number of DataLoader worker threads.
        epochs: Total number of training epochs.
        no_shuffle: If True, disables data shuffling even for train/val splits.

    Returns:
        Dataset namedtuple containing (set, loader, sampler).
    """
    dataset, loader, sampler = pcdet.datasets.build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=(split_name in ["train", "val"]) and not no_shuffle,
        total_epochs=epochs,
        seed=2345212,
        split=split_name
    )
    return Dataset(dataset, loader, sampler)


def build_model(
    cfg: EasyDict,
    train_data: Dataset
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any, Any]:
    """Instantiates network, moves it to CUDA, and builds optimizer and schedulers.

    Args:
        cfg: Model and optimization configuration.
        train_data: Dataset namedtuple containing training dataset and loader.

    Returns:
        Tuple of (model, optimizer, lr_scheduler, lr_warmup_scheduler).
    """
    model = pcdet.models.build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=train_data.set
    )
    model.cuda()
    optimizer = opt_utils.build_optimizer(
        model=model,
        optim_cfg=cfg.OPTIMIZATION
    )
    model.train()
    lr_scheduler, lr_warmup_scheduler = opt_utils.build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_data.loader),
        total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        last_epoch=-1,
        optim_cfg=cfg.OPTIMIZATION
    )
    return model, optimizer, lr_scheduler, lr_warmup_scheduler


# LOGGING UTILITIES


def create_default_logger(
    level: int = logging.INFO,
    name: Optional[str] = None
) -> Tuple[logging.Logger, logging.Formatter]:
    """Creates an empty logger and standard formatter to avoid duplicated handlers.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        name: Name of the logger instance.

    Returns:
        Tuple of (configured Logger, message Formatter).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.ERROR, "ERR")
    logging.addLevelName(logging.CRITICAL, "CRIT")
    formatter = logging.Formatter('%(asctime)s  %(levelname)4s  %(message)s', "%y-%m-%d %H:%M:%S")
    return logger, formatter


def get_console_handle(formatter: logging.Formatter) -> logging.StreamHandler:
    """Creates a console stream handler with the specified formatter.

    Args:
        formatter: Log message formatter.

    Returns:
        Configured StreamHandler.
    """
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    return console


def create_training_logger(
    output_path: Union[str, Path],
    filename: str = "train.log",
    console_only: bool = False,
    level: int = logging.INFO,
    name: Optional[str] = None
) -> logging.Logger:
    """Creates a logger logging both to console and a training log file.

    Args:
        output_path: Directory where the log file will be created.
        filename: Name of the output log file.
        console_only: If True, skips creating the file handler.
        level: Logging level.
        name: Name of the logger.

    Returns:
        Configured Logger instance.
    """
    logger, formatter = create_default_logger(level, name)
    console = get_console_handle(formatter)
    logger.addHandler(console)
    if console_only or output_path is None:
        return logger
    log_path = Path(output_path) / filename
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dummy_logger(name: Optional[str] = None) -> logging.Logger:
    """Creates a silent dummy logger with NullHandler.

    Args:
        name: Name of the logger instance.

    Returns:
        Logger instance that silences all messages.
    """
    logger, _ = create_default_logger(logging.INFO, name)
    logger.addHandler(logging.NullHandler())
    return logger


def create_console_logger(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Creates a simple console logger.

    Args:
        level: Logging level.
        name: Name of the logger instance.

    Returns:
        Configured Logger instance.
    """
    logger, formatter = create_default_logger(level, name)
    logger.addHandler(get_console_handle(formatter))
    return logger