import math
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class FourierPositionalEncoding(nn.Module):
    def __init__(self, input_dim: int = 2, encoding_dim: int = 128):
        super().__init__()
        if encoding_dim % (2 * input_dim) != 0:
            raise ValueError("encoding_dim must be divisible by 2 * input_dim")
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.register_buffer("frequencies", torch.linspace(0, 1, encoding_dim // (2 * input_dim)))

    def forward(self, x: Tensor) -> Tensor:
        encoded = x.unsqueeze(-1) * self.frequencies * (2 * math.pi)
        encoded = torch.cat([encoded.sin(), encoded.cos()], dim=-1)
        return encoded.flatten(start_dim=-2)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(64, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, feature_dim, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MultiScaleImageEncoder(nn.Module):
    def __init__(self, scales: Iterable[int] = (112, 224, 448), feature_dim: int = 256):
        super().__init__()
        self.scales = [int(s) for s in scales]
        self.encoders = nn.ModuleDict({str(s): ImageEncoder(feature_dim) for s in self.scales})

    def forward(self, images: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {str(s): self.encoders[str(s)](images[str(s)]) for s in self.scales}


class GeneConditioningEncoder(nn.Module):
    def __init__(self, gene_dim: int, condition_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, condition_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ExprPCAConditioner(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ScaleAwareFiLMLayer(nn.Module):
    def __init__(self, feature_dim: int = 256, condition_dim: int = 128, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.gamma = nn.ModuleList([nn.Linear(condition_dim, feature_dim) for _ in range(num_scales)])
        self.beta = nn.ModuleList([nn.Linear(condition_dim, feature_dim) for _ in range(num_scales)])
        self.scale_weights = nn.Parameter(torch.full((num_scales,), 1.0 / num_scales))

    def forward(self, features: Dict[str, Tensor], condition: Tensor) -> Tuple[Tensor, Tensor]:
        modulated = []
        for i, key in enumerate(features):
            modulated.append(self.gamma[i](condition) * features[key] + self.beta[i](condition))
        return torch.cat(modulated, dim=-1), self.scale_weights


class NeuralFieldMLPWithFiLM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 1000,
        condition_dim: int = 128,
        dropout: float = 0.2,
        enable_heg_head: bool = False,
    ):
        super().__init__()
        self.enable_heg_head = enable_heg_head
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.gamma = nn.ModuleList([nn.Linear(condition_dim, hidden_dim) for _ in self.layers])
        self.beta = nn.ModuleList([nn.Linear(condition_dim, hidden_dim) for _ in self.layers])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.heg_head = nn.Linear(hidden_dim, output_dim) if enable_heg_head else None

    def forward(self, x: Tensor, condition: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        for layer, gamma, beta in zip(self.layers, self.gamma, self.beta):
            x = layer(x)
            x = gamma(condition) * x + beta(condition)
            x = self.dropout(F.relu(x))
        main = torch.sigmoid(self.head(x)) * 14.0
        if self.heg_head is None:
            return main
        return main, torch.sigmoid(self.heg_head(x)) * 14.0


class MonteCarloSpotAggregator(nn.Module):
    def __init__(self, num_samples: int = 16, fourier_encoding: Optional[nn.Module] = None, noise_std: float = 0.1):
        super().__init__()
        self.num_samples = num_samples
        self.fourier_encoding = fourier_encoding
        self.noise_std = noise_std

    def forward(
        self,
        neural_field: nn.Module,
        image_features: Tensor,
        spot_centers: Tensor,
        condition: Tensor,
        spatial_pca: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size = spot_centers.size(0)
        n = self.num_samples
        noise = torch.randn(batch_size, n, 2, device=spot_centers.device) * self.noise_std
        points = spot_centers[:, None, :] + noise
        pos = self.fourier_encoding(points.reshape(-1, 2)) if self.fourier_encoding else points.reshape(-1, 2)

        img = image_features[:, None, :].expand(batch_size, n, -1).reshape(batch_size * n, -1)
        cond = condition[:, None, :].expand(batch_size, n, -1).reshape(batch_size * n, -1)
        parts = [img, pos]
        if spatial_pca is not None:
            parts.append(spatial_pca[:, None, :].expand(batch_size, n, -1).reshape(batch_size * n, -1))

        out = neural_field(torch.cat(parts, dim=-1), cond)
        if isinstance(out, tuple):
            main, heg = out
            return main.view(batch_size, n, -1).mean(dim=1), heg.view(batch_size, n, -1).mean(dim=1)
        return out.view(batch_size, n, -1).mean(dim=1), None


class HiFiST(nn.Module):
    def __init__(
        self,
        scales=(112, 224, 448),
        feature_dim: int = 256,
        condition_dim: int = 128,
        gene_dim: int = 1000,
        output_dim: int = 1000,
        fourier_dim: int = 128,
        use_fourier_pe: bool = True,
        enable_film: bool = True,
        uniform_scale_weights: bool = False,
        use_spatial_pca: bool = False,
        spatial_pca_dim: int = 2,
        use_expr_pca: bool = False,
        expr_pca_dim: int = 8,
        dropout_rate: float = 0.2,
        neural_hidden_dim: int = 512,
        aggregator_samples: int = 16,
        enable_heg_branch: bool = False,
    ):
        super().__init__()
        self.scales = [int(s) for s in scales]
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.num_scales = len(self.scales)
        self.use_spatial_pca = use_spatial_pca
        self.use_expr_pca = use_expr_pca
        self.enable_film = enable_film
        self.uniform_scale_weights = uniform_scale_weights

        self.image_encoder = MultiScaleImageEncoder(self.scales, feature_dim)
        self.gene_encoder = GeneConditioningEncoder(gene_dim, condition_dim)
        self.expr_conditioner = ExprPCAConditioner(expr_pca_dim, condition_dim) if use_expr_pca else None
        self.global_condition = nn.Parameter(torch.zeros(1, condition_dim))
        self.scale_aware_film = ScaleAwareFiLMLayer(feature_dim, condition_dim, self.num_scales)
        if uniform_scale_weights:
            self.scale_aware_film.scale_weights.requires_grad_(False)
        self.fourier_encoding = FourierPositionalEncoding(2, fourier_dim) if use_fourier_pe else None

        pos_dim = fourier_dim if use_fourier_pe else 2
        field_input_dim = feature_dim + pos_dim + (spatial_pca_dim if use_spatial_pca else 0)
        self.neural_field = NeuralFieldMLPWithFiLM(
            field_input_dim, neural_hidden_dim, output_dim, condition_dim, dropout_rate, enable_heg_branch
        )
        self.spot_aggregator = MonteCarloSpotAggregator(aggregator_samples, self.fourier_encoding)

    def _condition(self, batch_size: int, gene_expressions: Optional[Tensor], expr_pca: Optional[Tensor]) -> Tensor:
        condition = self.gene_encoder(gene_expressions) if gene_expressions is not None else self.global_condition.expand(batch_size, -1)
        if self.expr_conditioner is not None and expr_pca is not None:
            condition = condition + self.expr_conditioner(expr_pca)
        return condition

    def forward(
        self,
        multi_scale_images: Dict[str, Tensor],
        spot_centers: Tensor,
        gene_expressions: Optional[Tensor],
        spatial_pca: Optional[Tensor] = None,
        expr_pca: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        features = self.image_encoder(multi_scale_images)
        batch_size = next(iter(multi_scale_images.values())).size(0)
        condition = self._condition(batch_size, gene_expressions, expr_pca)

        if self.enable_film:
            stacked, scale_weights = self.scale_aware_film(features, condition)
        else:
            stacked = torch.cat([features[str(s)] for s in self.scales], dim=-1)
            scale_weights = torch.ones(self.num_scales, device=stacked.device) / self.num_scales

        if self.uniform_scale_weights:
            scale_weights = torch.ones_like(scale_weights) / self.num_scales

        aggregated = (
            stacked.view(-1, self.num_scales, self.feature_dim)
            * scale_weights[None, :, None]
        ).sum(dim=1)
        predictions, heg_predictions = self.spot_aggregator(
            self.neural_field,
            aggregated,
            spot_centers,
            condition,
            spatial_pca if self.use_spatial_pca else None,
        )
        return predictions, scale_weights, heg_predictions


def create_hifi_st_model(scales=(112, 224, 448), feature_dim=256, condition_dim=128, gene_dim=1000, output_dim=1000):
    return HiFiST(scales=scales, feature_dim=feature_dim, condition_dim=condition_dim, gene_dim=gene_dim, output_dim=output_dim)
