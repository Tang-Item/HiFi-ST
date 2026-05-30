from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import glob
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BasicTransform:
    def __init__(self, train: bool = True):
        self.train = train

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.train:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            angle = random.uniform(-15, 15)
            image = image.rotate(angle)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
        return (tensor - mean) / std


def build_transform(train: bool = True):
    return BasicTransform(train)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gene_list(*paths: Path) -> List[str]:
    for path in paths:
        if path.exists():
            return list(np.load(path, allow_pickle=True)) if path.suffix == ".npy" else [x.strip() for x in path.read_text().splitlines()]
    raise FileNotFoundError("No gene list found.")


class MultiScaleSpotDataset(data.Dataset):
    def __init__(
        self,
        spots_meta: pd.DataFrame,
        patch_centers: Dict[str, np.ndarray],
        patch_images_dir: str,
        gene_list: List[str],
        expression_matrices: Optional[Dict[str, np.ndarray]] = None,
        scales: Iterable[int] = (112, 224, 448),
        transform=None,
        train: bool = True,
        tissue_image_paths: Optional[Dict[str, str]] = None,
        spatial_pca_path: Optional[str] = None,
        expr_pca_path: Optional[str] = None,
        pca_ids_path: Optional[str] = None,
    ):
        self.spots_meta = spots_meta
        self.patch_centers = patch_centers
        self.patch_images_dir = Path(patch_images_dir)
        self.gene_list = gene_list
        self.expression_matrices = expression_matrices or {}
        self.scales = [int(s) for s in scales]
        self.transform = transform or build_transform(train)
        self.tissue_image_paths = tissue_image_paths or {}
        self.tissue_cache: Dict[str, Image.Image] = {}

        self.valid_spots = [
            str(sid) for sid in spots_meta.index
            if str(sid) in patch_centers and (self.patch_images_dir / f"{sid}.png").exists()
        ]
        self.spatial_pca = self._load_optional_embedding(spatial_pca_path, pca_ids_path)
        self.expr_pca = self._load_optional_embedding(expr_pca_path, pca_ids_path)
        print(f"MultiScaleSpotDataset: {len(self.valid_spots)} spots; scales={self.scales}")

    def __len__(self) -> int:
        return len(self.valid_spots)

    def __getitem__(self, idx: int):
        spot_id = self.valid_spots[idx]
        meta = self.spots_meta.loc[spot_id]
        sample_id = str(meta.get("sample_id", spot_id.split("_")[0]))
        coords = np.asarray([meta["x"], meta["y"]], dtype=np.float32)

        item = {
            "multi_scale_images": self._load_images(spot_id, sample_id, coords),
            "spot_centers": torch.from_numpy(coords),
            "gene_expressions": torch.from_numpy(self._load_expression(spot_id, sample_id)),
            "spot_id": spot_id,
            "patch_center": torch.as_tensor(self.patch_centers[spot_id], dtype=torch.float32),
        }
        if self.spatial_pca is not None:
            item["spatial_pca"] = torch.from_numpy(self.spatial_pca[idx])
        if self.expr_pca is not None:
            item["expr_pca"] = torch.from_numpy(self.expr_pca[idx])
        return item

    def _load_expression(self, spot_id: str, sample_id: str) -> np.ndarray:
        matrix = self.expression_matrices.get(sample_id)
        if matrix is None:
            return np.zeros(len(self.gene_list), dtype=np.float32)
        sample_spots = [sid for sid in self.spots_meta.index if str(sid).startswith(sample_id)]
        try:
            return matrix[:, sample_spots.index(spot_id)].astype(np.float32)
        except ValueError:
            return np.zeros(len(self.gene_list), dtype=np.float32)

    def _load_images(self, spot_id: str, sample_id: str, coords: np.ndarray) -> Dict[str, torch.Tensor]:
        patch = Image.open(self.patch_images_dir / f"{spot_id}.png").convert("RGB")
        images = {
            "224": self.transform(patch.resize((224, 224), Image.LANCZOS)),
            "112": self.transform(patch.resize((112, 112), Image.LANCZOS)),
        }
        if 448 in self.scales:
            images["448"] = self.transform(self._crop_tissue_patch(sample_id, coords, size=448))
        return {str(s): images.get(str(s), torch.zeros(3, s, s)) for s in self.scales}

    def _crop_tissue_patch(self, sample_id: str, coords: np.ndarray, size: int = 448) -> Image.Image:
        image = self._get_tissue_image(sample_id)
        if image is None:
            return Image.new("RGB", (size, size))
        x, y = map(int, coords)
        r = size // 2
        patch = image.crop((max(0, x - r), max(0, y - r), x + r, y + r))
        return patch if patch.size == (size, size) else patch.resize((size, size), Image.LANCZOS)

    def _get_tissue_image(self, sample_id: str) -> Optional[Image.Image]:
        if sample_id in self.tissue_cache:
            return self.tissue_cache[sample_id]
        path = self.tissue_image_paths.get(sample_id) or self._infer_tissue_image_path(sample_id)
        if path is None or not Path(path).exists():
            return None
        self.tissue_cache[sample_id] = Image.open(path).convert("RGB")
        return self.tissue_cache[sample_id]

    @staticmethod
    def _infer_tissue_image_path(sample_id: str) -> Optional[str]:
        candidates = []
        if sample_id:
            candidates += glob.glob(f"data/her2st/data/ST-imgs/{sample_id[0]}/{sample_id}/*.jpg")
        candidates += glob.glob(f"data/cSCC/GSE144240_RAW/*{sample_id}.jpg")
        candidates += [f"data/Alex_NatGen/Alex_NatGen/{sample_id}/{sample_id}/spatial/tissue_hires_image.png"]
        return next((p for p in candidates if Path(p).exists()), None)

    def _load_optional_embedding(self, arr_path: Optional[str], ids_path: Optional[str]) -> Optional[np.ndarray]:
        if arr_path is None or not Path(arr_path).exists():
            return None
        arr = np.load(arr_path).astype(np.float32)
        if ids_path and Path(ids_path).exists():
            ids = list(np.load(ids_path, allow_pickle=True))
            index = {str(sid): i for i, sid in enumerate(ids)}
            if all(sid in index for sid in self.valid_spots):
                return arr[[index[sid] for sid in self.valid_spots]]
            return None
        return arr if arr.shape[0] == len(self.valid_spots) else None


class HER2STDataset:
    def __init__(self, data_dir: str = "data/her2st"):
        self.data_dir = Path(data_dir)
        self.sample_names: List[str] = []

    def get_sample_names(self) -> List[str]:
        if self.sample_names:
            return self.sample_names
        spots_meta = read_pickle(self.data_dir / "data/HER2ST_spots_meta.pkl")
        names = sorted({str(s).split("_")[0] for s in spots_meta.index if str(s).split("_")[0] != "A1"})
        root = Path("data/preprocessed_expression_matrices/her2st")
        self.sample_names = [s for s in names if (root / s / "preprocessed_matrix.npy").exists()]
        return self.sample_names

    def load_data(self, fold: int = 0, train: bool = True):
        spots_meta = read_pickle(self.data_dir / "data/HER2ST_spots_meta.pkl")
        patch_centers = read_pickle(self.data_dir / "data/patch_centers.pkl")
        genes = load_gene_list(
            Path("data/preprocessed_expression_matrices/her2st/her2st_hvg_genes.npy"),
            self.data_dir / "data/HER2ST_genes.txt",
        )
        samples = self.get_sample_names()
        selected = [s for i, s in enumerate(samples) if (i != fold) == train]
        matrices = _load_expression_matrices(selected, Path("data/preprocessed_expression_matrices/her2st"))
        spots = [sid for sid in spots_meta.index if any(str(sid).startswith(s) for s in selected)]
        return MultiScaleSpotDataset(
            spots_meta.loc[spots], patch_centers, str(self.data_dir / "data/patch_images"), genes, matrices,
            train=train,
            spatial_pca_path="data/her2st/data/her2st_spatial_pca.npy",
            expr_pca_path="data/her2st/data/her2st_expr_pca.npy",
            pca_ids_path="data/her2st/data/her2st_spot_ids.npy",
        )


class CSCCDataset:
    def __init__(self, data_dir: str = "data/cSCC"):
        self.data_dir = Path(data_dir)
        self.sample_names: List[str] = []

    def get_sample_names(self) -> List[str]:
        if self.sample_names:
            return self.sample_names
        spots_meta = read_pickle(self.data_dir / "cSCC_spots_meta.pkl")
        names = {"_".join(str(s).split("_")[:3]) for s in spots_meta.index}
        self.sample_names = sorted(names, key=_cscc_sort_key)
        return self.sample_names

    def load_data(self, fold: int = 0, train: bool = True):
        spots_meta = read_pickle(self.data_dir / "cSCC_spots_meta.pkl")
        centers_path = self.data_dir / "patch_centers.pkl"
        patch_centers = read_pickle(centers_path) if centers_path.exists() else {
            sid: np.asarray([spots_meta.loc[sid, "x"], spots_meta.loc[sid, "y"]]) for sid in spots_meta.index
        }
        genes = load_gene_list(Path("data/preprocessed_expression_matrices/cscc/cscc_hvg_genes.npy"))
        samples = self.get_sample_names()
        selected = [s for i, s in enumerate(samples) if (i != fold) == train]
        matrices = _load_expression_matrices(selected, Path("data/preprocessed_expression_matrices/cscc"))
        spots = [sid for sid in spots_meta.index if any(str(sid).startswith(s) for s in selected)]
        dataset = MultiScaleSpotDataset(
            spots_meta.loc[spots], patch_centers, str(self.data_dir / "patch_images"), genes, matrices,
            train=train,
            spatial_pca_path="data/cSCC/cscc_spatial_pca.npy",
            expr_pca_path="data/cSCC/cscc_expr_pca.npy",
            pca_ids_path="data/cSCC/cscc_spot_ids.npy",
        )
        return dataset, genes


class Alex10xDataset:
    def __init__(self, alex_data_dir: str = "data/Alex_NatGen", tenx_data_dir: str = "data/10xGenomics"):
        self.alex_data_dir = Path(alex_data_dir)
        self.tenx_data_dir = Path(tenx_data_dir)
        self.sample_names: List[str] = []

    def get_sample_names(self) -> List[str]:
        if self.sample_names:
            return self.sample_names
        spots_meta = read_pickle(self.alex_data_dir / "Alex10x_spots_meta.pkl")
        self.sample_names = sorted({str(s).split("_")[0] for s in spots_meta.index if not str(s).split("_")[0].isdigit()})
        return self.sample_names

    def load_data(self, fold: int = 0, train: bool = True):
        spots_meta = read_pickle(self.alex_data_dir / "Alex10x_spots_meta.pkl")
        patch_centers = read_pickle(self.alex_data_dir / "Alex10x_patch_centers.pkl")
        genes = load_gene_list(Path("data/preprocessed_expression_matrices/alex10x/alex10x_hvg_genes.npy"))
        samples = self.get_sample_names()
        selected = [s for i, s in enumerate(samples) if (i != fold) == train]
        matrices = _load_expression_matrices(selected, Path("data/preprocessed_expression_matrices/alex10x"))
        spots = [sid for sid in spots_meta.index if any(str(sid).startswith(s) for s in selected)]
        return MultiScaleSpotDataset(
            spots_meta.loc[spots], patch_centers, str(self.alex_data_dir / "patch_images"), genes, matrices,
            train=train,
            spatial_pca_path="data/Alex_NatGen/alex10x_spatial_pca.npy",
            expr_pca_path="data/Alex_NatGen/alex10x_expr_pca.npy",
            pca_ids_path="data/Alex_NatGen/alex10x_spot_ids.npy",
        )


def _load_expression_matrices(samples: Iterable[str], root: Path) -> Dict[str, np.ndarray]:
    matrices = {}
    for sample in samples:
        path = root / sample / "preprocessed_matrix.npy"
        if path.exists():
            matrices[sample] = np.load(path)
    return matrices


def _cscc_sort_key(name: str):
    parts = name.split("_")
    patient = int(parts[0][1:]) if parts and parts[0].startswith("P") and parts[0][1:].isdigit() else 999
    rep = int(parts[2][3:]) if len(parts) > 2 and parts[2].startswith("rep") and parts[2][3:].isdigit() else 999
    return patient, rep


def collate_fn(batch):
    images = {}
    for item in batch:
        for scale, tensor in item["multi_scale_images"].items():
            images.setdefault(scale, []).append(tensor)
    result = {
        "multi_scale_images": {scale: torch.stack(tensors) for scale, tensors in images.items()},
        "spot_centers": torch.stack([x["spot_centers"] for x in batch]),
        "gene_expressions": torch.stack([x["gene_expressions"] for x in batch]),
        "spot_ids": [x["spot_id"] for x in batch],
        "patch_centers": torch.stack([x["patch_center"] for x in batch]),
    }
    for key in ("spatial_pca", "expr_pca"):
        if key in batch[0]:
            result[key] = torch.stack([x[key] for x in batch])
    return result


def create_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
