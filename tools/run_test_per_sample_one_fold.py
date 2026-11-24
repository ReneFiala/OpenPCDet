#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates per-sample metrics for a single fold pair (t, v) of nested cross-validation.
Outputs a pickle file named {method}-t{t}-v{v}.pkl.
"""

import argparse
import logging
from pathlib import Path
import pickle
from typing import Any, Dict
import warnings

from easydict import EasyDict
import scipy.stats
import torch

import culs_defs_eval as culs_eval
import culs_defs_loading as culs_loading
from pcdet.models import build_network


def get_results_for_cv(cfg: EasyDict, logger: logging.Logger, args: argparse.Namespace, t: int, v: int) -> Dict[str, Any]:
    """Evaluates per-sample metrics for the specified test/validation fold pair."""
    data_path = Path(args.data_root) / f"{args.method}-t{t}-v{v}"
    cfg.DATA_CONFIG.DATA_PATH = str(data_path)
    data = culs_loading.create_loaders(cfg, logger, "test")
    model_path = Path(args.checkpoint_root) / f"{args.method}-t{t}-v{v}" / args.checkpoint_name
    model = build_model(cfg, data.set, model_path)
    _, metrics = culs_eval.evaluate_model_scores_per_sample(cfg, model, data.loader)
    return metrics


def main(args: argparse.Namespace) -> None:
    """Runs evaluation for a single fold pair and saves result pickle."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
    torch.no_grad()
    torch.cuda.empty_cache()

    logger = culs_loading.create_console_logger()
    intro_print(args, logger)
    cfg = culs_loading.load_config(args.cfg, 0)
    result = get_results_for_cv(cfg, logger, args, args.t, args.v)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.method}-t{args.t}-v{args.v}.pkl", "wb") as f:
        pickle.dump(result, f)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments preserving exact original interface."""
    parser = argparse.ArgumentParser(description="Run per-sample evaluation on a single fold pair.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to base YAML config.")
    parser.add_argument("--method", type=str, required=True, help="Split method name (e.g. ci-3b).")
    parser.add_argument("--checkpoint-root", type=str, required=True, help="Root folder with checkpoint folders.")
    parser.add_argument("--checkpoint-name", type=str, required=True, help="Filename of checkpoint (.pth).")
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing dataset splits.")
    parser.add_argument("--out", type=str, required=True, help="Output directory path.")
    parser.add_argument("-t", type=int, required=True, help="Test fold index (0-4).")
    parser.add_argument("-v", type=int, required=True, help="Validation fold index (0-4).")
    return parser.parse_args()


def intro_print(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Logs evaluation parameters."""
    logger.info("Eval script provided arguments")
    logger.info(f"Config path:     {args.cfg}")
    logger.info(f"Split method:    {args.method}")
    logger.info(f"Model root:      {args.checkpoint_root}")
    logger.info(f"Model filename:  {args.checkpoint_name}")
    logger.info(f"Data root:       {args.data_root}")
    logger.info(f"Output dir:      {args.out}")
    logger.info(f"Test fold:       {args.t}")
    logger.info(f"Val fold:        {args.v}")


def build_model(cfg: EasyDict, dataset: Any, checkpoint: Path) -> torch.nn.Module:
    """Builds network and loads trained weights from checkpoint."""
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )
    model.load_params_from_file(
        filename=str(checkpoint),
        logger=logging.getLogger(__name__),
        map_location={"cuda:1": "cuda:0"},
        pre_trained_path=str(checkpoint)
    )
    model.cuda()
    return model


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)