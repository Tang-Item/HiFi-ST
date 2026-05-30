import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from dataset import Alex10xDataset, CSCCDataset, HER2STDataset, collate_fn
from model import HiFiST
from train import compute_metrics


class MclSTExpEvaluator:
    def __init__(self, model_path: str, device: torch.device, output_dir: str = "outputs"):
        self.model_path = Path(model_path)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_dataset_and_fold(self) -> Tuple[str, int]:
        name = self.model_path.parent.name
        if "_" not in name:
            return "HER2+", 0
        dataset, fold = name.rsplit("_", 1)
        return dataset, int(fold) if fold.isdigit() else 0

    def _load_dataset(self, dataset: str, fold: int):
        handlers = {"HER2+": HER2STDataset, "cSCC": CSCCDataset, "Alex+10x": Alex10xDataset}
        if dataset not in handlers:
            raise ValueError(f"Unknown dataset: {dataset}")
        val_dataset = handlers[dataset]().load_data(fold=fold, train=False)
        return val_dataset[0] if isinstance(val_dataset, tuple) else val_dataset

    def load_model(self, gene_dim: int, dataset_name: str, val_dataset=None) -> HiFiST:
        model = HiFiST(
            scales=[112, 224, 448],
            feature_dim=256,
            condition_dim=128,
            gene_dim=gene_dim,
            output_dim=gene_dim,
            use_spatial_pca=getattr(val_dataset, "spatial_pca", None) is not None,
            spatial_pca_dim=2,
            use_expr_pca=getattr(val_dataset, "expr_pca", None) is not None,
            expr_pca_dim=8,
            dropout_rate=0.2,
            neural_hidden_dim=512,
            aggregator_samples=16,
            enable_heg_branch=dataset_name == "cSCC",
        ).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def evaluate_dataset(self, dataset: str):
        dataset_from_path, fold = self._parse_dataset_and_fold()
        dataset = dataset_from_path or dataset
        val_dataset = self._load_dataset(dataset, fold)
        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False
        )
        model = self.load_model(len(val_dataset.gene_list), dataset, val_dataset)
        predictions, targets = [], []
        with torch.no_grad():
            for batch in loader:
                images = {k: v.to(self.device) for k, v in batch["multi_scale_images"].items()}
                centers = batch["spot_centers"].to(self.device)
                spatial_pca = batch.get("spatial_pca")
                expr_pca = batch.get("expr_pca")
                if spatial_pca is not None:
                    spatial_pca = spatial_pca.to(self.device)
                if expr_pca is not None:
                    expr_pca = expr_pca.to(self.device)
                pred, _, _ = model(images, centers, None, spatial_pca=spatial_pca, expr_pca=expr_pca)
                predictions.append(pred.cpu())
                targets.append(batch["gene_expressions"].cpu())
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()
        metrics = compute_metrics(predictions, targets)
        np.savez(self.output_dir / f"evaluation_results_{dataset}_{fold}.npz", predictions=predictions, targets=targets, **metrics)
        return metrics


if __name__ == "__main__":
    print("HiFi-ST evaluator")
