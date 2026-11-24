#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training callbacks for OpenPCDet: early stopping, metric logging (pickle, CSV),
and checkpoint management.
"""

from collections.abc import Iterable
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

from easydict import EasyDict
import train_utils.train_utils as train_utils


class CulsCallback:
    """Base callback class defining training event hooks."""

    def __init__(self) -> None:
        pass

    def on_epoch_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        train_predictions: List[Dict[str, Any]],
        train_metrics: Dict[str, Any],
        val_predictions: List[Dict[str, Any]],
        val_metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> bool:
        """Called at the end of each training epoch.

        Args:
            cfg: Global configuration dictionary.
            checkpoint: Model checkpoint state dictionary.
            epoch: Current epoch number.
            train_predictions: Predictions on the training set.
            train_metrics: Computed metrics for the training set.
            val_predictions: Predictions on the validation set.
            val_metrics: Computed metrics for the validation set.
            logger: Active logger instance.

        Returns:
            Boolean flag indicating whether training should stop early.
        """
        raise NotImplementedError()

    def on_train_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> None:
        """Called when the complete training process finishes.

        Args:
            cfg: Global configuration dictionary.
            checkpoint: Final model checkpoint state dictionary.
            epoch: Last executed epoch number.
            predictions: Predictions on the final evaluation set.
            metrics: Metrics computed for the final evaluation set.
            logger: Active logger instance.
        """
        raise NotImplementedError()

    def init_type_check(
        self,
        variables: List[Any],
        types: List[Union[type, Tuple[type, ...], List[type]]],
        names: List[str]
    ) -> None:
        """Validates variable types during callback initialization.

        Args:
            variables: List of variable values to check.
            types: Expected types or collections of allowed types.
            names: Variable names for error reporting.

        Raises:
            TypeError: If any variable does not match the expected type.
        """
        for v, t, n in zip(variables, types, names):
            if isinstance(t, (list, tuple)):
                good = any(isinstance(v, tt) for tt in t)
                if not good:
                    expected = [x.__name__ for x in t]
                    raise TypeError(f"Variable {n} is not any of {expected}")
                continue
            if not isinstance(v, t):
                raise TypeError(f"Variable {n} is not a {t.__name__}")


class EarlyStopCallback(CulsCallback):
    """Monitors validation metrics to stop training early and restore the best checkpoint."""

    def __init__(
        self,
        total_epochs: int,
        metric: str,
        objective: str,
        patience: int,
        warmup: int = 0,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """Initializes early stopping parameters.

        Args:
            total_epochs: Maximum planned epochs.
            metric: Target metric key to track (e.g. 'recall_sum_0.3').
            objective: Optimization goal ('min' or 'max').
            patience: Number of epochs without improvement before stopping.
            warmup: Number of initial warmup epochs where early stop is disabled.
            path: Directory path for exporting 'early_stop.pkl'.
        """
        super().__init__()
        self.init_type_check(
            [total_epochs, metric, objective, patience, warmup, path],
            [int, str, str, int, int, [str, Path]],
            ["total_epochs", "metric", "objective", "patience", "warmup", "path"]
        )
        self.total_epochs = total_epochs
        self.metric = metric
        self.objective = objective
        self.patience = patience
        self.best_value: Optional[float] = None
        self.wait: int = 0
        self.warmup = warmup
        self.path: Optional[Path] = Path(path) if path is not None else None
        self.best_epoch: int = 0
        self.best_checkpoint: Dict[str, Any] = {}
        self.best_metrics: Dict[str, Any] = {}
        self.best_predictions: List[Dict[str, Any]] = []
        self.dict: EasyDict = EasyDict()
        self.init_dict()

    def init_dict(self) -> None:
        """Initializes internal dictionary tracking early stopping parameters and state."""
        self.dict = EasyDict({
            "metric": self.metric,
            "objective": self.objective,
            "patience": self.patience,
            "warmup": self.warmup,
        })

    def on_epoch_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        train_predictions: List[Dict[str, Any]],
        train_metrics: Dict[str, Any],
        val_predictions: List[Dict[str, Any]],
        val_metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> bool:
        """Evaluates improvement on epoch finish and controls early stopping trigger."""
        if self.warmup >= epoch:
            self.save_dict()
            return False

        if self.metric not in val_metrics:
            logger.warning(f"EarlyStopCallback: Metric {self.metric} not found in metrics.")
            self.save_dict()
            return False

        metric = val_metrics[self.metric]

        if self.best_value is None:
            logger.debug(f"EarlyStopCallback: New metric {self.metric} with value {metric:.2f}.")
            self.update_best(checkpoint, epoch, val_predictions, val_metrics)
            self.save_dict()
            return False

        if self.is_metric_better(metric):
            self.update_best(checkpoint, epoch, val_predictions, val_metrics)
            logger.info(f"EarlyStopCallback: New best metric value {metric:.2f}.")
            self.save_dict()
            return False

        self.wait += 1
        stop = False

        if self.wait >= self.patience:
            logger.info(
                f"EarlyStopCallback: Patience reached, restoring checkpoint from epoch {self.best_epoch}."
            )
            stop = True

        if epoch == self.total_epochs and not self.is_metric_better(metric):
            logger.info(
                f"EarlyStopCallback: Maximum initial epochs reached, restoring checkpoint from epoch {self.best_epoch}."
            )
            stop = True

        if stop:
            checkpoint.clear()
            checkpoint.update(self.best_checkpoint)
            self.save_dict()
            return True

        logger.debug(f"EarlyStopCallback: No improvement. Waiting for {self.wait} epochs.")
        self.save_dict()
        return False

    def is_metric_better(self, metric: float) -> bool:
        """Checks if current metric value improves upon the recorded best value."""
        if self.best_value is None:
            return True
        if self.objective == "min" and metric < self.best_value:
            return True
        if self.objective == "max" and metric > self.best_value:
            return True
        return False

    def update_best(
        self,
        checkpoint: Dict[str, Any],
        epoch: int,
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> None:
        """Updates internal state with newly found optimal checkpoint and metrics."""
        self.best_value = metrics[self.metric]
        self.best_metrics = metrics
        self.best_predictions = predictions
        self.best_epoch = epoch
        self.best_checkpoint = checkpoint.copy()
        self.wait = 0
        self.dict.best_epoch = epoch
        self.dict.best_value = self.best_value

    def save_dict(self) -> None:
        """Persists early stopping state dictionary to disk as a pickle file."""
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        out_file = self.path / "early_stop.pkl"
        with open(out_file, "wb") as fp:
            pickle.dump(self.dict, fp)


class PickleMetricsCallback(CulsCallback):
    """Serializes epoch and final test metrics into a Python pickle file."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__()
        self.path = Path(path)
        self.output: Dict[str, Any] = {
            "train": {},
            "val": {}
        }

    def on_epoch_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        train_predictions: List[Dict[str, Any]],
        train_metrics: Dict[str, Any],
        val_predictions: List[Dict[str, Any]],
        val_metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> bool:
        """Records train and validation metrics for the current epoch and saves to disk."""
        self.output['train'][epoch] = train_metrics
        self.output['val'][epoch] = val_metrics
        self.save_file()
        return False

    def on_train_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> None:
        """Records final test metrics upon training completion and saves to disk."""
        self.output['test'] = metrics
        self.save_file()

    def save_file(self) -> None:
        """Dumps collected metrics dictionary into 'metrics.pkl'."""
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "metrics.pkl", "wb") as fp:
            pickle.dump(self.output, fp)


class CsvMetricsCallback(CulsCallback):
    """Logs epoch metrics and test evaluation results into separate CSV spreadsheets."""

    def __init__(
        self,
        path: Union[str, Path],
        sep: str = ",",
        append: bool = True,
        configs: Optional[Any] = None
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.written_epoch: bool = False
        self.written_test: bool = False
        self.sep = sep
        self.append = append

    def on_epoch_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        train_predictions: List[Dict[str, Any]],
        train_metrics: Dict[str, Any],
        val_predictions: List[Dict[str, Any]],
        val_metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> bool:
        """Logs train and validation epoch metric lines into CSV files."""
        self.update_file(epoch, train_metrics, "log_train.csv", include_epoch=True, written=self.written_epoch)
        self.update_file(epoch, val_metrics, "log_val.csv", include_epoch=True, written=self.written_epoch)
        self.written_epoch = True
        return False

    def on_train_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> None:
        """Logs final test metrics line into log_test.csv."""
        self.update_file(epoch, metrics, "log_test.csv", include_epoch=False, written=self.written_test)
        self.written_test = True

    def update_file(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        filename: str,
        include_epoch: bool,
        written: bool
    ) -> None:
        """Appends formatted metric line to CSV file, creating header if first time."""
        target_file = self.path / filename
        if not written:
            self.init_file(metrics, include_epoch, filename)
        with open(target_file, "a", encoding="utf-8") as fp:
            formatted_values = [
                ("n/a" if x is None else f"{x:.5f}")
                for x in metrics.values()
                if not isinstance(x, dict)
            ]
            if include_epoch:
                line = str(epoch) + self.sep + self.sep.join(formatted_values) + "\n"
            else:
                line = self.sep.join(formatted_values) + "\n"
            fp.write(line)

    def init_file(self, metrics: Dict[str, Any], include_epoch: bool, filename: str) -> None:
        """Prepares CSV header and verifies existing header compatibility if appending."""
        header_parts: List[str] = []
        if include_epoch:
            header_parts.append("epoch")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                header_parts.append(k)
        header = self.sep.join(header_parts) + "\n"

        self.path.mkdir(parents=True, exist_ok=True)
        target_file = self.path / filename

        if self.append and target_file.exists():
            with open(target_file, "r+", encoding="utf-8") as fp:
                lines = fp.readlines()
                if len(lines) == 0:
                    fp.write(header)
                    return
                if lines[0] != header:
                    raise IOError(f"CsvMetricsCallback: '{filename}' already exists and has a different header!")
        else:
            with open(target_file, "w", encoding="utf-8") as fp:
                fp.write(header)


class SaveLastCheckpointCallback(CulsCallback):
    """Saves final model checkpoint state upon training conclusion."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_train_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> None:
        """Saves current checkpoint to target path using OpenPCDet train_utils."""
        train_utils.save_checkpoint(checkpoint, self.path)


class SaveBestCheckpointsCallback(CulsCallback):
    """Tracks specified evaluation metrics and saves separate checkpoints for their best values."""

    def __init__(
        self,
        metrics_to_use: Iterable[Tuple[str, str]],
        path: Union[str, Path],
        save_checkpoints: bool = True,
        save_epoch_pkl: bool = True
    ) -> None:
        """Initializes tracked metric criteria.

        Args:
            metrics_to_use: Iterable of (metric_name, objective) pairs where objective is 'min' or 'max'.
            path: Directory path for saving checkpoints and bests summary.
            save_checkpoints: If True, exports model weights whenever an improvement occurs.
            save_epoch_pkl: If True, writes 'bests.pkl' dictionary.
        """
        super().__init__()
        metrics_error_msg = (
            "SaveBestCheckpointsCallback: metrics_to_use is in incorrect format.\n"
            "It must be an iterable of string pairs with metric names and metric objectives, "
            "e.g.: [('val_loss', 'min'), ('recall_0.5', 'max')]"
        )

        if not isinstance(metrics_to_use, Iterable):
            raise TypeError(metrics_error_msg)

        for metric in metrics_to_use:
            if len(metric) != 2:
                raise TypeError(metrics_error_msg)
            if not isinstance(metric[0], str) or not isinstance(metric[1], str):
                raise TypeError(metrics_error_msg)
            if metric[1] not in ["min", "max"]:
                raise TypeError(metrics_error_msg)

        self.metrics_cfg = [{"name": x[0], "goal": x[1]} for x in metrics_to_use]
        self.best_values: Dict[str, Dict[str, Optional[Union[float, int]]]] = {
            x[0]: {"val": None, "epoch": None} for x in metrics_to_use
        }
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_checkpoints = save_checkpoints
        self.save_epoch_pkl = save_epoch_pkl

    def on_epoch_done(
        self,
        cfg: EasyDict,
        checkpoint: Dict[str, Any],
        epoch: int,
        train_predictions: List[Dict[str, Any]],
        train_metrics: Dict[str, Any],
        val_predictions: List[Dict[str, Any]],
        val_metrics: Dict[str, Any],
        logger: logging.Logger
    ) -> bool:
        """Evaluates each tracked metric against its best value and triggers checkpoint save on improvement."""
        for metric_cfg in self.metrics_cfg:
            name = metric_cfg['name']
            if name not in val_metrics:
                continue
            current_value = val_metrics[name]
            if current_value is None:
                continue
            best_value = self.best_values[name]['val']

            if best_value is None:
                self.save_best(name, checkpoint, current_value, epoch)
                continue

            if (metric_cfg['goal'] == "min" and current_value < best_value) or \
               (metric_cfg['goal'] == "max" and current_value > best_value):
                self.save_best(name, checkpoint, current_value, epoch)
                continue

        if self.save_epoch_pkl:
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self.path / "bests.pkl", 'wb') as f:
                pickle.dump(self.best_values, f)
        return False

    def save_best(
        self,
        metric_name: str,
        checkpoint: Dict[str, Any],
        metric_value: float,
        epoch: int
    ) -> None:
        """Updates best tracked value and saves checkpoint file if enabled."""
        self.best_values[metric_name]['val'] = metric_value
        self.best_values[metric_name]['epoch'] = epoch
        filename = self.path / f"best_{metric_name}"
        if self.save_checkpoints:
            train_utils.save_checkpoint(checkpoint, filename)