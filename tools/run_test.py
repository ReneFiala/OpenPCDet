#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation CLI script for OpenPCDet models.
Loads model weights, evaluates metrics and exports predictions.pkl and metrics.pkl.
"""

import argparse
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Tuple
import warnings

from easydict import EasyDict
import scipy.stats
import torch

import culs_defs_eval as culs_eval
import culs_defs_loading as culs_loading
from pcdet.models import build_network


def main(args: argparse.Namespace) -> Tuple[Dict[str, Any], list]:
    """Runs evaluation over the test dataset split and saves outputs to disk.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (metrics dictionary, predictions list).
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
    torch.no_grad()
    torch.cuda.empty_cache()

    logger = culs_loading.create_console_logger()
    intro_print(args, logger)
    cfg = culs_loading.load_config(args.cfg, 0)

    if args.data is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.data

    if args.study is not None:
        with open(args.study, "rb") as fp:
            study = pickle.load(fp)
        culs_loading.load_trial_params(study, cfg)

    data = culs_loading.create_loaders(cfg, logger, "test")
    model = build_model(cfg, args, data.set)
    predictions, metrics = culs_eval.evaluate_model_scores(cfg, model, data.loader)

    if args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_predictions:
            with open(out_dir / "predictions.pkl", "wb") as f:
                pickle.dump(predictions, f)
        if not args.no_metrics:
            with open(out_dir / "metrics.pkl", "wb") as f:
                pickle.dump(metrics, f)

    return metrics, predictions


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments preserving exact original interface."""
    parser = argparse.ArgumentParser(description="Evaluate OpenPCDet model checkpoint on test data.")
    parser.add_argument("cfg", type=str, help="Path to model YAML configuration file.")
    parser.add_argument("checkpoint", type=str, help="Path to model weights checkpoint (.pth).")
    parser.add_argument("out", type=str, nargs="?", default=None, help="Output directory path.")
    parser.add_argument("--data", "-d", type=str, default=None, help="Alternative dataset directory path.")
    parser.add_argument("--study", "-s", type=str, default=None, help="Path to Optuna study pickle file.")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size per step.")
    parser.add_argument("--workers", "-w", type=int, default=1, help="DataLoader worker threads.")
    parser.add_argument("--no-metrics", action="store_true", help="Do not save metrics.pkl.")
    parser.add_argument("--no-predictions", action="store_true", help="Do not save predictions.pkl.")
    return parser.parse_args()


def intro_print(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Logs input arguments for the evaluation run."""
    logger.info("Eval script provided arguments")
    logger.info(f"Config path:     {args.cfg}")
    logger.info(f"Checkpoint path: {args.checkpoint}")
    logger.info(f"Output path:     {args.out}")
    logger.info(f"Dataset path:    {args.data}")
    logger.info(f"Study path:      {args.study}")
    logger.info(f"Batch size:      {args.batch_size}")
    logger.info(f"Workers:         {args.workers}")


def build_model(cfg: EasyDict, args: argparse.Namespace, dataset: Any) -> torch.nn.Module:
    """Instantiates network and loads checkpoint weights."""
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )
    model.load_params_from_file(
        filename=args.checkpoint,
        logger=logging.getLogger(__name__),
        map_location={"cuda:1": "cuda:0"},
        pre_trained_path=args.checkpoint
    )
    model.cuda()
    return model


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)