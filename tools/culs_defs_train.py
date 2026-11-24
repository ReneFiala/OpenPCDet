"""
Training loop, scheduler management, epoch stepping, and callback execution
for OpenPCDet models.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from easydict import EasyDict
import numpy as np
import pcdet.models
import torch
import train_utils.train_utils as train_utils

import culs_defs_callbacks as callbacks


def train_model(
    cfg: EasyDict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_normal_scheduler: Any,
    lr_warmup_scheduler: Any,
    logger: Any,
    train_data: Any,
    val_data: Optional[Any] = None,
    test_data: Optional[Any] = None,
    train_eval_func: Optional[Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = None,
    val_eval_func: Optional[Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = None,
    test_eval_func: Optional[Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = None,
    epoch_callbacks: Optional[List[callbacks.CulsCallback]] = None,
    train_callbacks: Optional[List[callbacks.CulsCallback]] = None,
    epochs_per_eval: int = 1
) -> Dict[str, Any]:
    """Executes the full multi-epoch training pipeline with evaluation hooks and callbacks.

    Args:
        cfg: Configuration dictionary.
        model: Detection network module.
        optimizer: PyTorch optimizer instance.
        lr_normal_scheduler: Primary learning rate scheduler.
        lr_warmup_scheduler: Warmup learning rate scheduler.
        logger: Active logger instance.
        train_data: Dataset namedtuple for training data.
        val_data: Optional dataset namedtuple for validation.
        test_data: Optional dataset namedtuple for final test.
        train_eval_func: Optional function to evaluate model on the training set.
        val_eval_func: Optional function to evaluate model on the validation set.
        test_eval_func: Optional function to evaluate model on the test set.
        epoch_callbacks: Callbacks invoked after each epoch.
        train_callbacks: Callbacks invoked when training completes.
        epochs_per_eval: Frequency (in epochs) of evaluation runs.

    Returns:
        Dictionary containing final checkpoint state and collected metric lists.
    """
    if epoch_callbacks is None:
        epoch_callbacks = []
    if train_callbacks is None:
        train_callbacks = []
    if epochs_per_eval < 1:
        raise ValueError(f"epochs_per_eval must be a positive non-zero integer, got {epochs_per_eval}.")

    type_check(callbacks.CulsCallback, epoch_callbacks)
    type_check(callbacks.CulsCallback, train_callbacks)

    # Standard OpenPCDet forward-pass wrapper computing predictions and losses
    train_func = pcdet.models.model_fn_decorator()
    accumulated_iter = 0
    train_metrics: List[Dict[str, Any]] = []
    val_metrics: List[Dict[str, Any]] = []
    checkpoint: Dict[str, Any] = {}
    shifted_epoch: int = 0

    for current_epoch in range(cfg.OPTIMIZATION.NUM_EPOCHS):
        shifted_epoch = current_epoch + 1
        start_time = time.perf_counter()

        if train_data.sampler is not None:
            train_data.sampler.set_epoch(current_epoch)

        lr_scheduler = choose_scheduler(
            lr_normal_scheduler, lr_warmup_scheduler, cfg, current_epoch
        )

        accumulated_iter, train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_data.loader,
            model_func=train_func,
            lr_scheduler=lr_scheduler,
            accumulated_iter=accumulated_iter,
            cfg=cfg,
            cur_epoch=current_epoch
        )

        checkpoint = train_utils.checkpoint_state(model, optimizer, current_epoch, accumulated_iter)
        tr_time = time.perf_counter()
        tr_dur = tr_time - start_time

        if train_eval_func and current_epoch % epochs_per_eval == 0:
            epoch_train_result, epoch_train_metrics = train_eval_func(cfg, model, train_data.loader)
            epoch_train_metrics["epoch_time"] = tr_dur
        else:
            epoch_train_result, epoch_train_metrics = {}, {}

        if val_eval_func and current_epoch % epochs_per_eval == 0 and val_data is not None:
            epoch_val_result, epoch_val_metrics = val_eval_func(cfg, model, val_data.loader)
            epoch_val_metrics["epoch_time"] = tr_dur
        else:
            epoch_val_result, epoch_val_metrics = {}, {}

        ev_time = time.perf_counter()
        ev_dur = ev_time - tr_time

        train_metrics.append(epoch_train_metrics)
        val_metrics.append(epoch_val_metrics)

        stop_early = False
        for cb in epoch_callbacks:
            if cb.on_epoch_done(
                cfg, checkpoint, shifted_epoch,
                epoch_train_result, epoch_train_metrics,
                epoch_val_result, epoch_val_metrics, logger
            ):
                stop_early = True

        cb_time = time.perf_counter()
        cb_dur = cb_time - ev_time

        total_epoch_time = tr_dur + ev_dur + cb_dur
        logger.info(
            f"Epoch {str(shifted_epoch).zfill(3)}  "
            f"Time: {tr_dur:.1f}/{ev_dur:.1f}/{cb_dur:.1f} ({total_epoch_time:.1f})"
        )

        if stop_early:
            break

    if test_eval_func and test_data is not None:
        test_result, test_metrics = test_eval_func(cfg, model, test_data.loader)
    else:
        test_result, test_metrics = {}, {}

    for cb in train_callbacks:
        cb.on_train_done(cfg, checkpoint, shifted_epoch, test_result, test_metrics, logger)

    return {
        "checkpoint": checkpoint,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }


def type_check(parent: type, children: List[Any]) -> None:
    """Verifies that all elements in a list inherit from a parent class.

    Args:
        parent: Expected base class type.
        children: List of instances to validate.

    Raises:
        TypeError: If input is not a list or an element is not an instance of parent.
    """
    if not isinstance(children, list):
        raise TypeError(f"{type(children).__name__} is not a list.")
    for child in children:
        if not isinstance(child, parent):
            raise TypeError(f"{type(child).__name__} is not a {parent.__name__} type.")


def choose_scheduler(normal_lr: Any, warmup_lr: Any, cfg: EasyDict, epoch: int) -> Any:
    """Selects warmup or normal learning rate scheduler based on current epoch.

    Args:
        normal_lr: Primary learning rate scheduler.
        warmup_lr: Warmup learning rate scheduler.
        cfg: Optimization configuration dictionary.
        epoch: Current epoch index.

    Returns:
        Active scheduler instance for the current epoch.
    """
    if warmup_lr is not None and epoch < cfg.OPTIMIZATION.WARMUP_EPOCH:
        return warmup_lr
    return normal_lr


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    model_func: Callable[..., Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]],
    lr_scheduler: Any,
    accumulated_iter: int,
    cfg: EasyDict,
    cur_epoch: int
) -> Tuple[int, float]:
    """Iterates through one epoch of training batches and updates model weights.

    Args:
        model: Detection network module.
        optimizer: Optimizer instance.
        train_loader: DataLoader yielding training batches.
        model_func: Forward function computing loss and metrics.
        lr_scheduler: Active learning rate scheduler.
        accumulated_iter: Global step counter.
        cfg: Configuration dictionary.
        cur_epoch: Current epoch index.

    Returns:
        Tuple of (updated global iteration count, average epoch loss).
    """
    model.train()
    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)
    optim_cfg = cfg.OPTIMIZATION
    start_it = accumulated_iter % total_it_each_epoch
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
        init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16)
    )
    losses: List[float] = []

    for _ in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        lr_scheduler.step(accumulated_iter, cur_epoch)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=False):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        accumulated_iter += 1

    avg_loss = float(np.average(losses)) if losses else 0.0
    return accumulated_iter, avg_loss