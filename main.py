import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from dataset import Alex10xDataset, CSCCDataset, HER2STDataset, create_dataloader
from eval import MclSTExpEvaluator
from train import train_hifi_st


DATASETS = {"HER2+": HER2STDataset, "cSCC": CSCCDataset, "Alex+10x": Alex10xDataset}


def parse_args():
    parser = argparse.ArgumentParser(description="HiFi-ST")
    parser.add_argument("--dataset", default="cSCC", choices=list(DATASETS))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--scales", nargs="+", type=int, default=[112, 224, 448])
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--condition_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--aggregator_samples", type=int, default=16)
    parser.add_argument("--disable_fourier_pe", action="store_true")
    parser.add_argument("--disable_film", action="store_true")
    parser.add_argument("--uniform_scale_weights", action="store_true")
    parser.add_argument("--disable_heg_branch", action="store_true")
    parser.add_argument("--use_expr_pca", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def load_dataset(dataset: str, fold: int):
    handler = DATASETS[dataset]()
    train_set = handler.load_data(fold=fold, train=True)
    val_set = handler.load_data(fold=fold, train=False)
    if isinstance(train_set, tuple):
        train_set = train_set[0]
    if isinstance(val_set, tuple):
        val_set = val_set[0]
    return train_set, val_set, val_set


def run_train(args, fold: int):
    train_set, val_set, test_set = load_dataset(args.dataset, fold)
    train_loader = create_dataloader(train_set, args.batch_size, True, args.num_workers)
    val_loader = create_dataloader(val_set, args.batch_size, False, args.num_workers)
    test_loader = create_dataloader(test_set, args.batch_size, False, args.num_workers)
    output_dir = Path(args.output_dir) / f"{args.dataset}_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return train_hifi_st(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        device=get_device(args.device),
        gene_dim=len(train_set.gene_list),
        dataset_name=args.dataset,
        dropout_rate=args.dropout_rate,
        scales=args.scales,
        feature_dim=args.feature_dim,
        condition_dim=args.condition_dim,
        neural_hidden_dim=args.hidden_dim,
        aggregator_samples=args.aggregator_samples,
        use_fourier_pe=not args.disable_fourier_pe,
        enable_film=not args.disable_film,
        uniform_scale_weights=args.uniform_scale_weights,
        enable_heg_branch=not args.disable_heg_branch and args.dataset == "cSCC",
        use_expr_pca=args.use_expr_pca or getattr(train_set, "expr_pca", None) is not None,
        lr=args.lr,
    )


def run_eval(args):
    if not args.model_path:
        args.model_path = str(Path(args.output_dir) / f"{args.dataset}_{args.fold}" / "best_model.pth")
    evaluator = MclSTExpEvaluator(args.model_path, get_device(args.device), str(Path(args.output_dir) / f"evaluation_{args.dataset}_{args.fold}"))
    return evaluator.evaluate_dataset(args.dataset)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.evaluate:
        print(run_eval(args))
        return

    folds = range(len(DATASETS[args.dataset]().get_sample_names())) if args.all_folds else [args.fold]
    for fold in folds:
        print(f"Training {args.dataset} fold {fold}")
        run_train(args, fold)


if __name__ == "__main__":
    main()
