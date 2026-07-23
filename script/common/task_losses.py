"""Supervised task losses for the PAVR ClinVar demo."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


def class_counts_from_labels(
    labels: Union[torch.Tensor, np.ndarray, Sequence[int]],
    num_classes: int,
) -> np.ndarray:
    """Count labels per class.

    Parameters
    ----------
    labels:
        Integer class indices.
    num_classes:
        Number of classes ``C`` (output length). Missing classes receive count 0.

    Returns
    -------
    np.ndarray
        Float counts of shape ``[C]``.
    """
    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy().astype(np.int64)
    else:
        y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    return counts[:num_classes]


def logit_adjustment_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_counts: Union[torch.Tensor, np.ndarray, Sequence[float]],
    tau: float = 1.0,
) -> torch.Tensor:
    """Class-prior adjusted cross-entropy for imbalanced labels.

    Implements ``CE(logits + τ log π, y)`` following Menon et al. (ICML 2021),
    where ``π`` is the empirical class prior estimated from ``class_counts``.

    Parameters
    ----------
    logits:
        Unnormalized scores ``[B, C]``.
    target:
        Integer targets ``[B]``.
    class_counts:
        Training-set counts used to form ``π``.
    tau:
        Temperature on the log-prior adjustment.
    """
    if isinstance(class_counts, torch.Tensor):
        counts = class_counts.to(dtype=logits.dtype, device=logits.device)
    else:
        counts = torch.tensor(class_counts, dtype=logits.dtype, device=logits.device)
    prior = counts / counts.sum().clamp_min(1.0)
    log_prior = torch.log(prior.clamp_min(1.0e-12))
    adjusted = logits + float(tau) * log_prior
    return F.cross_entropy(adjusted, target.long())
