"""PAVR package exports.

Public API for the paper-default Perturbation-Aware Variant Representation
adapter (dual-branch encoder + classification head).
"""
from pavr.model import (
    AlleleBranch,
    BackboneNormalization,
    ContextBranch,
    MultiViewPerturbation,
    PAVRClassifier,
    PAVRConfig,
    PAVRLossConfig,
    PAVRModel,
    PAVRObjective,
    SoftComplementaryWindow,
    attention_exclusivity_loss,
    pavr_probe_features,
)

__all__ = [
    "AlleleBranch",
    "BackboneNormalization",
    "ContextBranch",
    "MultiViewPerturbation",
    "PAVRClassifier",
    "PAVRConfig",
    "PAVRLossConfig",
    "PAVRModel",
    "PAVRObjective",
    "SoftComplementaryWindow",
    "attention_exclusivity_loss",
    "pavr_probe_features",
]
