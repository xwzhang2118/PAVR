"""PAVR: Perturbation-Aware Variant Representation.

Paper-default adapter over frozen genomic foundation model (GFM) tokens.

Pipeline
--------
1. BackboneNormalization — map heterogeneous GFM dims to a shared space
2. MultiViewPerturbation — build token-wise ref/alt perturbation views
3. SoftComplementaryWindow — learn a soft allele/context radius via decay-MLP
4. AlleleBranch + ContextBranch — local allelic vs broader contextual response
5. Probe — concatenate [z_allele, z_context, |z_allele - z_context|]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import GlobalAttentionPool, ProjectionHead, _masked_logits, masked_mean_pool


@dataclass
class PAVRConfig:
    """Hyper-parameters for the PAVR dual-branch adapter.

    Attributes
    ----------
    backbone_dim:
        Hidden size of the frozen GFM token embeddings (last dim of ``H_ref`` / ``H_alt``).
    common_dim:
        Shared width after backbone normalization.
    projection_dim:
        Output width of each branch projection head.
    distance_dim:
        Width of distance-to-variant and variant-type embeddings.
    local_radius:
        Initial soft-window radius (in tokens) used to bias the decay-MLP.
    num_variant_types:
        Size of the discrete variant-type embedding table.
    dropout:
        Dropout rate inside projections and branch mixers.
    attention_hidden_dim:
        Hidden size of additive attention scoring; defaults to ``common_dim``.
    radius_min, radius_max:
        Clamp range for the predicted soft-window radius ``r``.
    radius_temperature:
        Temperature ``T`` in ``w = sigmoid((r - d) / T)``.
    decay_mlp_hidden:
        Hidden width of the MLP that maps the distance-binned decay profile to ``r``.
    context_complement_alpha:
        Strength of the context complement: ``w_out = 1 - alpha * w_in``.
    window_eps:
        Numerical floor for window masses and logs.
    """

    backbone_dim: int
    common_dim: int = 256
    projection_dim: int = 128
    distance_dim: int = 16
    local_radius: int = 32
    num_variant_types: int = 8
    dropout: float = 0.1
    attention_hidden_dim: Optional[int] = None
    radius_min: float = 4.0
    radius_max: float = 128.0
    radius_temperature: float = 2.0
    decay_mlp_hidden: int = 32
    context_complement_alpha: float = 1.0
    window_eps: float = 1.0e-6


class BackboneNormalization(nn.Module):
    """Linear + GELU + LayerNorm projection shared by ``H_ref`` and ``H_alt``.

    Stabilizes scale differences across GFMs before building the perturbation field.
    """

    def __init__(self, backbone_dim: int, common_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, common_dim),
            nn.GELU(),
            nn.LayerNorm(common_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h_ref: torch.Tensor, h_alt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project paired token tensors of shape ``[B, L, backbone_dim]``."""
        if h_ref.shape != h_alt.shape:
            raise ValueError(f"h_ref and h_alt must have same shape, got {h_ref.shape} and {h_alt.shape}")
        if h_ref.ndim != 3:
            raise ValueError(f"h_ref and h_alt must be [B, L, D], got {tuple(h_ref.shape)}")
        return self.proj(h_ref), self.proj(h_alt)


class DistanceToVariantEmbedding(nn.Module):
    """Embed absolute token distance to the variant center into discrete buckets.

    The variant center is the mean position of ``variant_mask`` (fallback: sequence
    mid-point when the mask is empty). Distances are bucketized and looked up in a
    learned embedding table.
    """

    def __init__(self, distance_dim: int, boundaries: Sequence[int] = (0, 1, 4, 16, 64, 256)):
        super().__init__()
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32), persistent=False)
        self.embedding = nn.Embedding(len(boundaries) + 1, distance_dim)

    def forward(
        self,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(distance_emb, distance, variant_center)``.

        Parameters
        ----------
        variant_mask:
            Boolean / float mask ``[B, L]`` marking variant-centered tokens.
        padding_mask:
            Optional ``[B, L]`` mask where ``True`` marks padding.

        Returns
        -------
        distance_emb:
            ``[B, L, distance_dim]`` bucket embeddings.
        distance:
            ``[B, L]`` absolute distances to the variant center.
        variant_center:
            ``[B]`` continuous center positions.
        """
        if variant_mask.ndim != 2:
            raise ValueError(f"variant_mask must be [B, L], got {tuple(variant_mask.shape)}")
        device = variant_mask.device
        batch_size, seq_len = variant_mask.shape
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        weights = variant_mask.to(device=device, dtype=torch.float32)
        if padding_mask is not None:
            valid = ~padding_mask.to(device=device, dtype=torch.bool)
            weights = weights * valid.to(dtype=weights.dtype)
        center = (positions * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        no_variant = weights.sum(dim=1) <= 0
        if no_variant.any():
            if padding_mask is None:
                fallback_center = positions[:, seq_len // 2]
            else:
                valid = ~padding_mask.to(device=device, dtype=torch.bool)
                fallback_center = (positions * valid.to(dtype=positions.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            center = torch.where(no_variant, fallback_center, center)

        distance = (positions - center.unsqueeze(1)).abs()
        bucket = torch.bucketize(distance, self.boundaries.to(device=device), right=True)
        distance_emb = self.embedding(bucket.clamp_max(self.embedding.num_embeddings - 1))
        return distance_emb, distance, center


class SoftComplementaryWindow(nn.Module):
    """Soft complementary allele / context window with a learnable decay-MLP radius.

    For each sample the module:

    1. Aggregates token perturbation strengths ``s_i = ||Δ_i||`` into distance bins
       to form a decay profile ``u``.
    2. Predicts radius ``r = softplus(MLP(u)) + radius_min`` (clamped).
    3. Defines soft masses ``w_in = sigmoid((r - d) / T)`` and
       ``w_out = 1 - α · w_in``.

    Allele attention is biased by ``log w_in``; context attention by ``log w_out``.
    """

    def __init__(
        self,
        boundaries: Sequence[int] = (0, 1, 4, 16, 64, 256),
        hidden_dim: int = 32,
        radius_min: float = 4.0,
        radius_max: float = 128.0,
        temperature: float = 2.0,
        complement_alpha: float = 1.0,
        init_radius: float = 32.0,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        if not (0.0 < complement_alpha <= 1.0):
            raise ValueError(f"complement_alpha must be in (0, 1], got {complement_alpha}")
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.temperature = float(max(temperature, eps))
        self.complement_alpha = float(complement_alpha)
        self.eps = float(eps)
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32), persistent=False)
        n_bins = len(boundaries) + 1
        self.mlp = nn.Sequential(
            nn.Linear(n_bins, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize so softplus(bias) + radius_min ≈ init_radius for a near-uniform profile.
        target = max(float(init_radius) - self.radius_min, 1.0e-3)
        init_bias = math.log(math.expm1(target))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(
        self,
        delta_h: torch.Tensor,
        distance: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute complementary window masses and log-biases.

        Parameters
        ----------
        delta_h:
            Token-wise ``H_alt - H_ref`` of shape ``[B, L, D]``.
        distance:
            Absolute distances ``[B, L]`` to the variant center.
        padding_mask:
            Optional padding mask ``[B, L]`` (``True`` = pad).

        Returns
        -------
        dict
            ``window_inside``, ``window_outside``, ``window_radius``,
            ``decay_profile``, ``log_inside``, ``log_outside``.
        """
        if delta_h.ndim != 3:
            raise ValueError(f"delta_h must be [B, L, D], got {tuple(delta_h.shape)}")
        if distance.shape != delta_h.shape[:2]:
            raise ValueError("distance must match delta_h on [B, L].")

        strength = delta_h.norm(dim=-1).to(dtype=torch.float32)
        dist = distance.to(device=delta_h.device, dtype=torch.float32)
        valid = torch.ones_like(strength, dtype=torch.bool)
        if padding_mask is not None:
            valid = ~padding_mask.to(device=delta_h.device, dtype=torch.bool)
        valid_f = valid.to(dtype=strength.dtype)

        n_bins = int(self.boundaries.numel()) + 1
        bucket = torch.bucketize(dist, self.boundaries.to(device=dist.device), right=True)
        bucket = bucket.clamp(0, n_bins - 1)
        batch_size = strength.shape[0]
        sum_s = strength.new_zeros((batch_size, n_bins))
        counts = strength.new_zeros((batch_size, n_bins))
        sum_s.scatter_add_(1, bucket, strength * valid_f)
        counts.scatter_add_(1, bucket, valid_f)
        profile = sum_s / counts.clamp_min(1.0)
        # L1-normalize so the MLP sees profile shape, not absolute GFM scale.
        profile = profile / profile.sum(dim=1, keepdim=True).clamp_min(self.eps)

        radius = F.softplus(self.mlp(profile).squeeze(-1)) + self.radius_min
        radius = radius.clamp(min=self.radius_min, max=self.radius_max)

        inside = torch.sigmoid((radius.unsqueeze(1) - dist) / self.temperature)
        inside = inside.clamp(min=self.eps, max=1.0)
        outside = (1.0 - self.complement_alpha * inside).clamp(min=self.eps, max=1.0)
        inside = inside * valid_f
        outside = outside * valid_f
        return {
            "window_inside": inside,
            "window_outside": outside,
            "window_radius": radius,
            "decay_profile": profile,
            "log_inside": inside.clamp_min(self.eps).log(),
            "log_outside": outside.clamp_min(self.eps).log(),
        }


class MultiViewPerturbation(nn.Module):
    """Construct the token-level multi-view perturbation field.

    For each token ``i``::

        T_i = [H_ref_i, H_alt_i, Δ_i, |Δ_i|, H_ref_i ⊙ H_alt_i,
               e_dist(i), e_type(t)]

    where ``Δ = H_alt - H_ref``, ``e_dist`` is the distance embedding, and
    ``e_type`` is the variant-type embedding broadcast over the sequence.
    """

    def __init__(self, common_dim: int, distance_dim: int, num_variant_types: int):
        super().__init__()
        self.common_dim = common_dim
        self.distance = DistanceToVariantEmbedding(distance_dim)
        self.variant_type_embedding = nn.Embedding(num_variant_types, distance_dim)
        # five content views + distance embedding + type embedding
        self.feature_dim = common_dim * 5 + distance_dim * 2

    def forward(
        self,
        h_ref: torch.Tensor,
        h_alt: torch.Tensor,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        variant_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return token features and auxiliary geometric tensors."""
        if h_ref.shape != h_alt.shape:
            raise ValueError(f"h_ref and h_alt must have same shape, got {h_ref.shape} and {h_alt.shape}")
        if h_ref.shape[:2] != variant_mask.shape:
            raise ValueError("h_ref/h_alt and variant_mask must agree on [B, L].")

        batch_size, seq_len, _ = h_ref.shape
        delta = h_alt - h_ref
        distance_emb, distance, variant_center = self.distance(variant_mask, padding_mask=padding_mask)

        if variant_type is None:
            variant_type = torch.zeros(batch_size, device=h_ref.device, dtype=torch.long)
        variant_type = variant_type.to(device=h_ref.device, dtype=torch.long).clamp(
            min=0,
            max=self.variant_type_embedding.num_embeddings - 1,
        )
        variant_emb = self.variant_type_embedding(variant_type).unsqueeze(1).expand(-1, seq_len, -1)
        token_features = torch.cat(
            [
                h_ref,
                h_alt,
                delta,
                delta.abs(),
                h_ref * h_alt,
                distance_emb.to(dtype=h_ref.dtype),
                variant_emb.to(dtype=h_ref.dtype),
            ],
            dim=-1,
        )

        if padding_mask is not None:
            valid = (~padding_mask.to(device=h_ref.device, dtype=torch.bool)).unsqueeze(-1).to(token_features.dtype)
            token_features = token_features * valid
            delta = delta * valid

        return {
            "token_features": token_features,
            "delta_h": delta,
            "distance": distance,
            "variant_center": variant_center,
            "variant_type": variant_type,
        }


class AlleleBranch(nn.Module):
    """Allele branch: local depthwise conv mixer + soft inside-window attention.

    Captures allele-local disruption around the variant site. Attention logits are
    additively biased by ``log_inside`` from :class:`SoftComplementaryWindow`.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_variant_types: int,
        dropout: float = 0.1,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        pad = kernel_size // 2
        self.dw = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=hidden_dim,
            bias=False,
        )
        self.pw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.variant_query = nn.Embedding(num_variant_types, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.projection = ProjectionHead(hidden_dim, projection_dim, dropout)

    def forward(
        self,
        token_features: torch.Tensor,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        variant_type: Optional[torch.Tensor] = None,
        log_inside: Optional[torch.Tensor] = None,
        window_inside: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return allele pooled features ``h_allele``, projected ``z_allele``, and attention."""
        if token_features.ndim != 3:
            raise ValueError(f"token_features must be [B, L, D], got {tuple(token_features.shape)}")
        if token_features.shape[:2] != variant_mask.shape:
            raise ValueError("token_features and variant_mask must agree on [B, L].")

        batch_size = token_features.shape[0]
        if variant_type is None:
            variant_type = torch.zeros(batch_size, device=token_features.device, dtype=torch.long)
        variant_type = variant_type.to(device=token_features.device, dtype=torch.long).clamp(
            min=0,
            max=self.variant_query.num_embeddings - 1,
        )

        x = self.input_proj(token_features)
        # Depthwise-separable local mixer (motif-scale).
        y = self.pw(F.gelu(self.dw(x.transpose(1, 2))))
        if y.shape[-1] != x.shape[1]:
            y = y[..., : x.shape[1]]
        encoded = self.norm(x + self.dropout(y.transpose(1, 2)))

        query = self.variant_query(variant_type)
        logits = self.score(torch.tanh(encoded + query.unsqueeze(1))).squeeze(-1)
        if log_inside is not None:
            logits = logits + log_inside.to(device=logits.device, dtype=logits.dtype)

        keep_mask = torch.ones_like(logits, dtype=torch.bool)
        if padding_mask is not None:
            keep_mask &= ~padding_mask.to(device=token_features.device, dtype=torch.bool)
        # Keep variant tokens eligible even if the soft window mass is tiny.
        keep_mask |= variant_mask.to(device=token_features.device, dtype=torch.bool)
        logits = _masked_logits(logits, keep_mask)
        attention = torch.softmax(logits, dim=-1)
        h_allele = (encoded * attention.unsqueeze(-1)).sum(dim=1)
        z_allele = self.projection(h_allele)

        if window_inside is None:
            inside_mass = attention.sum(dim=1)
        else:
            inside_mass = (attention * window_inside.to(device=attention.device, dtype=attention.dtype)).sum(dim=1)
        variant_mass = (attention * variant_mask.to(device=attention.device, dtype=attention.dtype)).sum(dim=1)
        return {
            "h_allele": h_allele,
            "z_allele": z_allele,
            "allele_attention": attention,
            "allele_inside_mass": inside_mass,
            "allele_variant_mass": variant_mass,
        }


class ContextBranch(nn.Module):
    """Context branch: bidirectional GRU over the full field + outside-window attention.

    Models longer-range contextual responses complementary to the allele branch.
    Attention logits are biased by ``log_outside`` from :class:`SoftComplementaryWindow`.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        attention_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional GRU.")
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = GlobalAttentionPool(hidden_dim, attention_hidden_dim)
        self.projection = ProjectionHead(hidden_dim, projection_dim, dropout)

    def forward(
        self,
        token_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        log_outside: Optional[torch.Tensor] = None,
        window_outside: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return context pooled features ``h_context``, projected ``z_context``, and attention."""
        x = self.input_proj(token_features)
        encoded, _ = self.encoder(x)
        if padding_mask is not None:
            valid = (~padding_mask.to(device=encoded.device, dtype=torch.bool)).unsqueeze(-1).to(encoded.dtype)
            encoded = encoded * valid
        pooled = self.pool(encoded, padding_mask=padding_mask, logit_bias=log_outside)
        h_context = pooled["pooled"]
        z_context = self.projection(h_context)
        attention = pooled["attention"]
        if window_outside is not None:
            outside_mass = (attention * window_outside.to(device=attention.device, dtype=attention.dtype)).sum(dim=1)
        else:
            outside_mass = attention.new_ones(attention.shape[0])
        return {
            "h_context": h_context,
            "z_context": z_context,
            "context_attention": attention,
            "context_encoded": encoded,
            "context_outside_mass": outside_mass,
        }


def pavr_probe_features(z_allele: torch.Tensor, z_context: torch.Tensor) -> torch.Tensor:
    """Build the default supervised probe features.

    Returns the concatenation ``[z_allele, z_context, |z_allele - z_context|]``
    of shape ``[B, 3 * projection_dim]``.
    """
    return torch.cat([z_allele, z_context, (z_allele - z_context).abs()], dim=-1)


class PAVRModel(nn.Module):
    """Full PAVR adapter over frozen GFM token embeddings.

    Inputs are paired reference / alternate hidden states. Outputs include branch
    representations, soft-window diagnostics, and the concatenated probe vector used
    by the downstream classifier.
    """

    def __init__(self, config: PAVRConfig):
        super().__init__()
        self.config = config
        attention_hidden_dim = config.attention_hidden_dim or config.common_dim
        self.backbone_norm = BackboneNormalization(config.backbone_dim, config.common_dim, config.dropout)
        self.perturbation = MultiViewPerturbation(
            common_dim=config.common_dim,
            distance_dim=config.distance_dim,
            num_variant_types=config.num_variant_types,
        )
        self.window = SoftComplementaryWindow(
            hidden_dim=config.decay_mlp_hidden,
            radius_min=config.radius_min,
            radius_max=config.radius_max,
            temperature=config.radius_temperature,
            complement_alpha=config.context_complement_alpha,
            init_radius=float(config.local_radius),
            eps=config.window_eps,
        )
        self.allele_branch = AlleleBranch(
            feature_dim=self.perturbation.feature_dim,
            hidden_dim=config.common_dim,
            projection_dim=config.projection_dim,
            num_variant_types=config.num_variant_types,
            dropout=config.dropout,
        )
        self.context_branch = ContextBranch(
            feature_dim=self.perturbation.feature_dim,
            hidden_dim=config.common_dim,
            projection_dim=config.projection_dim,
            attention_hidden_dim=attention_hidden_dim,
            dropout=config.dropout,
        )

    def _norm_stats(
        self,
        delta_h: torch.Tensor,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Log1p summary statistics of perturbation magnitudes (diagnostic tensor)."""
        token_norm = delta_h.norm(dim=-1)
        valid = torch.ones_like(token_norm, dtype=torch.bool)
        if padding_mask is not None:
            valid = ~padding_mask.to(device=delta_h.device, dtype=torch.bool)
        valid_float = valid.to(dtype=delta_h.dtype)
        mean_norm = (token_norm * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        max_norm = token_norm.masked_fill(~valid, 0.0).max(dim=1).values
        var_mask = variant_mask.to(device=delta_h.device, dtype=torch.bool) & valid
        var_float = var_mask.to(dtype=delta_h.dtype)
        allele_norm = (token_norm * var_float).sum(dim=1) / var_float.sum(dim=1).clamp_min(1.0)
        pooled_delta_norm = masked_mean_pool(delta_h, padding_mask=padding_mask).norm(dim=-1)
        stats = torch.stack([mean_norm, max_norm, allele_norm, pooled_delta_norm], dim=-1)
        return torch.log1p(stats)

    def forward(
        self,
        h_ref: torch.Tensor,
        h_alt: torch.Tensor,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        variant_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the PAVR encoder.

        Parameters
        ----------
        h_ref, h_alt:
            Frozen GFM token states ``[B, L, backbone_dim]``.
        variant_mask:
            Variant-centered token mask ``[B, L]``.
        padding_mask:
            Optional padding mask ``[B, L]`` (``True`` = pad).
        variant_type:
            Optional discrete variant-type ids ``[B]``.

        Returns
        -------
        dict
            Includes ``z_allele``, ``z_context``, ``probe_features``, soft-window
            tensors, and attention maps from both branches.
        """
        h_ref_norm, h_alt_norm = self.backbone_norm(h_ref, h_alt)
        perturb = self.perturbation(
            h_ref_norm,
            h_alt_norm,
            variant_mask=variant_mask,
            padding_mask=padding_mask,
            variant_type=variant_type,
        )
        norm_stats = self._norm_stats(perturb["delta_h"], variant_mask=variant_mask, padding_mask=padding_mask)
        window = self.window(
            delta_h=perturb["delta_h"],
            distance=perturb["distance"],
            padding_mask=padding_mask,
        )
        allele = self.allele_branch(
            perturb["token_features"],
            variant_mask=variant_mask,
            padding_mask=padding_mask,
            variant_type=perturb["variant_type"],
            log_inside=window["log_inside"],
            window_inside=window["window_inside"],
        )
        context = self.context_branch(
            perturb["token_features"],
            padding_mask=padding_mask,
            log_outside=window["log_outside"],
            window_outside=window["window_outside"],
        )
        z_allele = allele["z_allele"]
        z_context = context["z_context"]
        probe_features = pavr_probe_features(z_allele, z_context)
        z_effect = F.normalize((z_allele + z_context) / 2.0, dim=-1)
        out: Dict[str, torch.Tensor] = {
            "h_ref_norm": h_ref_norm,
            "h_alt_norm": h_alt_norm,
            "delta_h": perturb["delta_h"],
            "distance": perturb["distance"],
            "variant_center": perturb["variant_center"],
            "norm_stats": norm_stats,
            "z_allele": z_allele,
            "z_context": z_context,
            "z_effect": z_effect,
            "probe_features": probe_features,
            "window_inside": window["window_inside"],
            "window_outside": window["window_outside"],
            "window_radius": window["window_radius"],
            "decay_profile": window["decay_profile"],
            "attention": context["context_attention"],
        }
        out.update(allele)
        out.update(context)
        return out


@dataclass
class PAVRLossConfig:
    """Weights for PAVR representation losses.

    Paper default uses attention exclusivity only
    (``attention_excl_weight = 0.05``).
    """

    attention_excl_weight: float = 0.05


def attention_exclusivity_loss(
    allele_attn: torch.Tensor,
    context_attn: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize overlapping allele and context attention mass.

    Computes the mean of ``sum_i a_i · c_i`` after optional padding renormalization,
    encouraging the two branches to focus on complementary positions.
    """
    a = allele_attn.clamp_min(1.0e-12)
    c = context_attn.clamp_min(1.0e-12)
    if padding_mask is not None:
        valid = (~padding_mask.to(device=a.device, dtype=torch.bool)).to(dtype=a.dtype)
        a = a * valid
        c = c * valid
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        c = c / c.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return (a * c).sum(dim=1).mean()


class PAVRObjective(nn.Module):
    """PAVR encoder wrapped with the paper-default representation loss.

    The nested attribute is named ``encoder`` so released demo checkpoints remain
    loadable without remapping state-dict keys.
    """

    def __init__(
        self,
        backbone_dim: int,
        common_dim: int = 256,
        projection_dim: int = 128,
        distance_dim: int = 16,
        local_radius: int = 32,
        num_variant_types: int = 8,
        dropout: float = 0.1,
        radius_min: float = 4.0,
        radius_max: float = 128.0,
        radius_temperature: float = 2.0,
        decay_mlp_hidden: int = 32,
        context_complement_alpha: float = 1.0,
        loss_config: Optional[PAVRLossConfig] = None,
    ):
        super().__init__()
        self.encoder = PAVRModel(
            PAVRConfig(
                backbone_dim=backbone_dim,
                common_dim=common_dim,
                projection_dim=projection_dim,
                distance_dim=distance_dim,
                local_radius=local_radius,
                num_variant_types=num_variant_types,
                dropout=dropout,
                radius_min=radius_min,
                radius_max=radius_max,
                radius_temperature=radius_temperature,
                decay_mlp_hidden=decay_mlp_hidden,
                context_complement_alpha=context_complement_alpha,
            )
        )
        self.loss_config = loss_config or PAVRLossConfig()

    def forward(
        self,
        h_ref: torch.Tensor,
        h_alt: torch.Tensor,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        variant_type: Optional[torch.Tensor] = None,
        gene_id: Optional[object] = None,
        allele_frequency: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward encoder and attach ``loss_repr`` / ``loss_attention_excl``."""
        del gene_id, allele_frequency  # reserved for optional extensions; unused by default
        outputs = self.encoder(
            h_ref=h_ref,
            h_alt=h_alt,
            variant_mask=variant_mask,
            padding_mask=padding_mask,
            variant_type=variant_type,
        )
        loss_excl = h_ref.new_zeros(())
        if self.loss_config.attention_excl_weight > 0:
            loss_excl = attention_exclusivity_loss(
                outputs["allele_attention"],
                outputs["context_attention"],
                padding_mask=padding_mask,
            )
        loss_repr = self.loss_config.attention_excl_weight * loss_excl
        outputs["loss_attention_excl"] = loss_excl
        outputs["loss_repr"] = loss_repr
        return outputs


class PAVRClassifier(nn.Module):
    """End-to-end PAVR classifier for multi-class pathogenicity prediction.

    Combines :class:`PAVRObjective` with an MLP head over
    ``[z_allele, z_context, |z_a - z_c|]``.
    """

    def __init__(
        self,
        backbone_dim: int,
        num_classes: int = 4,
        common_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 128,
        num_variant_types: int = 8,
        local_radius: int = 32,
        radius_temperature: float = 2.0,
        attention_excl_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = PAVRObjective(
            backbone_dim=backbone_dim,
            common_dim=common_dim,
            projection_dim=projection_dim,
            num_variant_types=num_variant_types,
            local_radius=local_radius,
            radius_temperature=radius_temperature,
            loss_config=PAVRLossConfig(attention_excl_weight=attention_excl_weight),
        )
        probe_dim = projection_dim * 3
        self.head = nn.Sequential(
            nn.Linear(probe_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        h_ref: torch.Tensor,
        h_alt: torch.Tensor,
        variant_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        variant_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return encoder outputs plus ``prediction`` / ``logits`` of shape ``[B, C]``."""
        out = self.encoder(
            h_ref=h_ref,
            h_alt=h_alt,
            variant_mask=variant_mask,
            padding_mask=padding_mask,
            variant_type=variant_type,
        )
        out["prediction"] = self.head(out["probe_features"])
        out["logits"] = out["prediction"]
        return out
