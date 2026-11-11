import argparse, logging, pickle, warnings
import scipy, torch
import culs_defs_eval as culs_eval
import culs_defs_loading as culs_loading
from pcdet.models import build_network
from pathlib import Path
from easydict import EasyDict

HARDCODED = False
args = EasyDict({
    "cfg": "./cfgs/base_second_mdl.yaml",
    "checkpoint_root": "./results/2026-07-01_grouped_second/ci-3b",
    "checkpoint_name": "best_ap_sum_0.3.pth",
    "out": "./results/doesthiswork",
    "data_root": "../data",
    "method": "ci-3b",
})


def get_results_for_cv(cfg: EasyDict, logger, args: argparse.Namespace, t, v):
    data_path = Path(args.data_root) / f"{args.method}-t{t}-v{v}"
    cfg.DATA_CONFIG.DATA_PATH = data_path
    data = culs_loading.create_loaders(cfg, logger, "test")
    model_path = Path(args.checkpoint_root) / f"{args.method}-t{t}-v{v}" / args.checkpoint_name
    model = build_model(cfg, data.set, model_path)
    predictions, metrics = culs_eval.evaluate_model_scores_per_sample(cfg, model, data.loader)
    return metrics


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning) # PyTorch
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
    torch.no_grad()
    torch.cuda.empty_cache()
    
    logger = culs_loading.create_console_logger()
    intro_print(args, logger)
    cfg = culs_loading.load_config(args.cfg, 0)
    results = {}
    for t in range(5):
        for v in range(5):
            if t == v:
                continue
            cv_result = get_results_for_cv(cfg, logger, args, t, v)
            results[f"t{t}-v{v}"] = cv_result
    with open(Path(args.out) / f"{args.method}.pkl", 'wb') as f:
        pickle.dump(results, f)


def print_metrics(metrics):
    keys = list(metrics.keys())
    keys.sort()
    key_length = max(len(x) for x in keys)
    for k in keys:
        if isinstance(metrics[k], dict) or isinstance(metrics[k], list):
            continue
        print(f"{k.ljust(key_length)}: {metrics[k]:.3f}")
        
def print_two_metrics(a, b):
    keys = list(a.keys())
    keys.sort()
    key_length = max(len(x) for x in keys)
    for k in keys:
        if isinstance(a[k], dict) or isinstance(a[k], list):
            continue
        print(f"{k.ljust(key_length)}: {a[k]:.3f} {b[k]:.3f} {'!!!' if a[k] != b[k] else ''}")

"""
   "cfg": "./cfgs/base_second_mdl.yaml",
    "checkpoint_root": "./results/2026-07-01_grouped_second/ci-3b",
    "checkpoint_name": "best_ap_sum_0.3.pth",
    "out": "./results/doesthiswork",
    "data_root": "../data",
    "method": "ci-3b",

"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--checkpoint-root', type=str)
    parser.add_argument('--checkpoint-name', type=str)
    parser.add_argument('--data-root', type=str)
    parser.add_argument('--out', type=str)
    return parser.parse_args()

def intro_print(args, logger):
    logger.info("Eval script provided arguments")
    logger.info(f"Config path:     {args.cfg}")
    logger.info(f"Split method:    {args.method}")
    logger.info(f"Model root:      {args.checkpoint_root}")
    logger.info(f"Model filename:  {args.checkpoint_name}")
    logger.info(f"Data root:       {args.data_root}")
    logger.info(f"Output dir:      {args.out}")

def build_model(cfg, dataset, checkpoint):
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )
    model.load_params_from_file(
        filename=checkpoint,
        logger=logging.getLogger(__name__),
        map_location={'cuda:1': 'cuda:0'},
        pre_trained_path=checkpoint
    )
    model.cuda()
    return model


if __name__ == "__main__":
    if not HARDCODED:
        args = parse_args()
    main(args)
