"""Training utilities for HiFi-ST."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import HiFiST


def _as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _safe_pcc(x, y, eps: float = 1e-8) -> Optional[float]:
    if np.std(x) <= eps or np.std(y) <= eps:
        return None
    r, _ = pearsonr(x, y)
    return None if np.isnan(r) else float(np.clip(r, -1.0, 1.0))


def compute_metrics(predictions, targets) -> Dict[str, float]:
    predictions = np.clip(np.nan_to_num(_as_numpy(predictions), nan=0.0, posinf=14.0, neginf=0.0), 0.0, 14.0)
    targets = np.clip(np.nan_to_num(_as_numpy(targets), nan=0.0, posinf=14.0, neginf=0.0), 0.0, 14.0)
    pred_log, targ_log = np.log1p(predictions), np.log1p(targets)

    gene_pccs = [_safe_pcc(predictions[:, i], targets[:, i]) for i in range(predictions.shape[1])]
    gene_pccs = np.asarray([r for r in gene_pccs if r is not None])

    sample_pccs = [_safe_pcc(predictions[i], targets[i]) for i in range(predictions.shape[0])]
    sample_pccs = np.asarray([r for r in sample_pccs if r is not None])

    top_k = min(50, predictions.shape[1])
    top_idx = np.argsort(targets.mean(axis=0))[::-1][:top_k]
    top_pccs = [_safe_pcc(predictions[:, i], targets[:, i]) for i in top_idx]
    top_pccs = np.asarray([r for r in top_pccs if r is not None])

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    return {
        "MSE": float(mean_squared_error(targ_log, pred_log)),
        "MAE": float(mean_absolute_error(targ_log, pred_log)),
        "MSE (Raw)": float(mean_squared_error(targets, predictions)),
        "MAE (Raw)": float(mean_absolute_error(targets, predictions)),
        "R²": float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0,
        "Overall PCC": _safe_pcc(predictions.ravel(), targets.ravel()) or 0.0,
        "Gene Mean PCC": float(gene_pccs.mean()) if gene_pccs.size else 0.0,
        "Gene Median PCC": float(np.median(gene_pccs)) if gene_pccs.size else 0.0,
        "Sample Mean PCC": float(sample_pccs.mean()) if sample_pccs.size else 0.0,
        "Sample Median PCC": float(np.median(sample_pccs)) if sample_pccs.size else 0.0,
        "Top 50 HEG Mean PCC": float(top_pccs.mean()) if top_pccs.size else 0.0,
        "Genes with PCC > 0.5 (%)": float((gene_pccs > 0.5).mean() * 100) if gene_pccs.size else 0.0,
        "Genes with PCC > 0.7 (%)": float((gene_pccs > 0.7).mean() * 100) if gene_pccs.size else 0.0,
        "Samples with PCC > 0.5 (%)": float((sample_pccs > 0.5).mean() * 100) if sample_pccs.size else 0.0,
        "Samples with PCC > 0.7 (%)": float((sample_pccs > 0.7).mean() * 100) if sample_pccs.size else 0.0,
    }


class HiFiSTLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, pcc_weight: float = 0.5, scale_weight: float = 0.01, heg_weight: float = 0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.pcc_weight = pcc_weight
        self.scale_weight = scale_weight
        self.heg_weight = heg_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, model=None, scale_weights=None, heg_predictions=None):
        pred_log = torch.log1p(torch.clamp(predictions, 0, 14))
        targ_log = torch.log1p(torch.clamp(targets, 0, 14))
        mse_loss = self.mse(pred_log, targ_log)
        pcc_loss = -self._pearson(pred_log.flatten(), targ_log.flatten())
        scale_loss = self._scale_loss(scale_weights)
        heg_loss = self._heg_loss(heg_predictions if heg_predictions is not None else predictions, targets)
        total = self.mse_weight * mse_loss + self.pcc_weight * pcc_loss + self.scale_weight * scale_loss + self.heg_weight * heg_loss
        return total, {
            "mse_loss": float(mse_loss.detach().cpu()),
            "pcc_loss": float(pcc_loss.detach().cpu()),
            "scale_loss": float(scale_loss.detach().cpu()),
            "heg_loss": float(heg_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }

    @staticmethod
    def _pearson(x, y, eps: float = 1e-8):
        x, y = x - x.mean(), y - y.mean()
        return (x * y).sum() / (torch.sqrt((x * x).sum() + eps) * torch.sqrt((y * y).sum() + eps) + eps)

    @staticmethod
    def _scale_loss(scale_weights):
        if scale_weights is None:
            return torch.tensor(0.0)
        return ((scale_weights - 1.0 / scale_weights.numel()) ** 2).mean()

    def _heg_loss(self, predictions, targets):
        if self.heg_weight <= 0:
            return predictions.new_tensor(0.0)
        top_k = min(50, targets.shape[1])
        top_idx = torch.topk(targets.mean(dim=0), k=top_k).indices
        pred = torch.log1p(torch.clamp(predictions[:, top_idx], 0, 14))
        targ = torch.log1p(torch.clamp(targets[:, top_idx], 0, 14))
        return self.mse(pred, targ) - self._pearson(pred.flatten(), targ.flatten())


class HiFiSTTrainer:
    def __init__(
        self,
        model: HiFiST,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "outputs",
        lr: float = 5e-5,
        weight_decay: float = 1e-5,
        patience: int = 30,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = criterion or HiFiSTLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=8)
        self.patience = patience
        self.best_pcc = -float("inf")
        self.best_state = None

    def train(self, num_epochs: int = 50):
        log_path = self.output_dir / "training_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,gene_mean_pcc,top50_heg_pcc,mse,mae\n")

        stale_epochs = 0
        for epoch in range(num_epochs):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            preds, targets = self.predict(self.val_loader)
            metrics = compute_metrics(preds, targets)
            pcc = metrics["Gene Mean PCC"]
            self.scheduler.step(pcc)

            if pcc > self.best_pcc:
                self.best_pcc = pcc
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_state, self.output_dir / "best_model.pth")
                stale_epochs = 0
            else:
                stale_epochs += 1

            line = f"{epoch},{train_loss:.6f},{val_loss:.6f},{pcc:.6f},{metrics['Top 50 HEG Mean PCC']:.6f},{metrics['MSE']:.6f},{metrics['MAE']:.6f}\n"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
            print(line.strip())

            if stale_epochs >= self.patience:
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        torch.save(self.model.state_dict(), self.output_dir / "final_model.pth")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        losses = []
        iterator = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in iterator:
            batch = self._to_device(batch)
            with torch.set_grad_enabled(train):
                preds, scale_weights, heg_preds = self.model(
                    batch["multi_scale_images"],
                    batch["spot_centers"],
                    None,
                    spatial_pca=batch.get("spatial_pca"),
                    expr_pca=batch.get("expr_pca"),
                )
                loss, _ = self.criterion(preds, batch["gene_expressions"], self.model, scale_weights, heg_preds)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="predict", leave=False):
                batch = self._to_device(batch)
                out, _, _ = self.model(
                    batch["multi_scale_images"],
                    batch["spot_centers"],
                    None,
                    spatial_pca=batch.get("spatial_pca"),
                    expr_pca=batch.get("expr_pca"),
                )
                preds.append(out.cpu())
                targets.append(batch["gene_expressions"].cpu())
        return torch.cat(preds).numpy(), torch.cat(targets).numpy()

    def evaluate(self, test_loader: DataLoader):
        preds, targets = self.predict(test_loader)
        metrics = compute_metrics(preds, targets)
        np.savez(self.output_dir / "evaluation_results.npz", predictions=preds, targets=targets, **metrics)
        return metrics

    def _to_device(self, batch):
        out = dict(batch)
        out["multi_scale_images"] = {k: v.to(self.device, non_blocking=True) for k, v in batch["multi_scale_images"].items()}
        for key in ("spot_centers", "gene_expressions", "patch_centers", "spatial_pca", "expr_pca"):
            if key in out and isinstance(out[key], torch.Tensor):
                out[key] = out[key].to(self.device, non_blocking=True)
        return out


def train_hifi_st(
    train_loader,
    val_loader,
    test_loader=None,
    output_dir="outputs",
    num_epochs=50,
    device=None,
    gene_dim=2000,
    dataset_name=None,
    dropout_rate: float = None,
    enable_log_calibration: bool = True,
    enable_staged_lr: bool = False,
    mse_loss_weight: float = None,
    pcc_loss_weight: float = None,
    training_stage: str = "stage2",
    scales=None,
    feature_dim: int = 256,
    condition_dim: int = 128,
    neural_hidden_dim: int = None,
    aggregator_samples: int = None,
    use_fourier_pe: bool = True,
    enable_film: bool = True,
    uniform_scale_weights: bool = False,
    enable_heg_branch: bool = None,
    use_expr_pca: bool = None,
    heg_pcc_weight_override: float = None,
    heg_mse_weight_override: float = None,
    heg_cosine_weight_override: float = None,
    minimal_outputs: bool = False,
    lr: float = 5e-5,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales = scales or [112, 224, 448]
    dropout_rate = 0.2 if dropout_rate is None else dropout_rate
    neural_hidden_dim = neural_hidden_dim or 512
    aggregator_samples = aggregator_samples or 16
    use_spatial_pca = getattr(train_loader.dataset, "spatial_pca", None) is not None
    use_expr_pca = (getattr(train_loader.dataset, "expr_pca", None) is not None) if use_expr_pca is None else use_expr_pca
    enable_heg_branch = (dataset_name or "").lower() == "cscc" if enable_heg_branch is None else enable_heg_branch

    model = HiFiST(
        scales=scales,
        feature_dim=feature_dim,
        condition_dim=condition_dim,
        gene_dim=gene_dim,
        output_dim=gene_dim,
        use_spatial_pca=use_spatial_pca,
        spatial_pca_dim=2,
        use_expr_pca=use_expr_pca,
        expr_pca_dim=8,
        dropout_rate=dropout_rate,
        neural_hidden_dim=neural_hidden_dim,
        aggregator_samples=aggregator_samples,
        enable_heg_branch=enable_heg_branch,
        use_fourier_pe=use_fourier_pe,
        enable_film=enable_film,
        uniform_scale_weights=uniform_scale_weights,
    )
    criterion = HiFiSTLoss(
        mse_weight=1.0 if mse_loss_weight is None else mse_loss_weight,
        pcc_weight=0.5 if pcc_loss_weight is None else pcc_loss_weight,
        heg_weight=1.0 if enable_heg_branch else 0.0,
    )
    trainer = HiFiSTTrainer(model, train_loader, val_loader, device, output_dir, lr=lr, criterion=criterion)
    trainer.train(num_epochs=num_epochs)
    metrics = trainer.evaluate(test_loader) if test_loader is not None else None
    return trainer, metrics


if __name__ == "__main__":
    print("HiFi-ST training module")
