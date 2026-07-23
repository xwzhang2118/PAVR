#!/usr/bin/env python3
"""Train PAVR on cached Evo2 token features (ClinVar gastric, 1 kb window).

Example (from repo root ``PAVR/``):

    PYTHONPATH=script python script/train.py \\
        --features data/evo2_7b_window_gastric_1k.pt \\
        --csv data/clinvar_pathogenicity.csv \\
        --output-dir result/rerun_window_gastric_1k \\
        --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.task_losses import class_counts_from_labels, logit_adjustment_cross_entropy  # noqa: E402
from pavr.model import PAVRClassifier  # noqa: E402

# ClinVar four-class pathogenicity labels used by the demo CSV.
CLASSES = ["Benign", "Likely_benign", "Likely_pathogenic", "Pathogenic"]

# Map free-text / ClinVar consequence strings to discrete variant-type ids.
VARIANT_TYPE_TO_INDEX = {
    "unknown": 0,
    "single_nucleotide_variant": 1,
    "snv": 1,
    "snp": 1,
    "deletion": 2,
    "insertion": 3,
    "indel": 4,
    "duplication": 5,
    "inversion": 6,
    "complex": 7,
    "microsatellite": 7,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for demo training on cached GFM features."""
    p = argparse.ArgumentParser(description="Train PAVR on cached GFM token features.")
    p.add_argument("--features", type=Path, default=ROOT / "data" / "evo2_7b_window_gastric_1k.pt",
                   help="Path to frozen token feature cache (.pt).")
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "clinvar_pathogenicity.csv",
                   help="Variant table with labels and train/val/test splits.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "result" / "rerun_window_gastric_1k",
                   help="Directory for checkpoints, predictions, and summary.json.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Random seeds for repeated runs.")
    p.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    p.add_argument("--patience", type=int, default=4, help="Early-stopping patience on validation AUROC.")
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--common-dim", type=int, default=256, help="Shared width after BackboneNorm.")
    p.add_argument("--projection-dim", type=int, default=128, help="Branch projection width.")
    p.add_argument("--hidden-dim", type=int, default=128, help="Classifier MLP hidden width.")
    p.add_argument("--local-radius", type=int, default=32, help="Initial soft-window radius (tokens).")
    p.add_argument("--radius-temperature", type=float, default=2.0, help="Soft-window sigmoid temperature.")
    p.add_argument("--attention-excl-weight", type=float, default=0.05,
                   help="Weight of allele/context attention exclusivity loss.")
    p.add_argument("--imbalance-strategy", default="logit_adjustment", choices=["none", "logit_adjustment"],
                   help="Supervised CE variant under class imbalance.")
    p.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    p.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True,
                   help="Use inverse-frequency WeightedRandomSampler on the training split.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variant_type_index(value: object) -> int:
    """Map a free-text variant type / consequence string to an integer id."""
    key = str(value).strip().lower()
    if key in {"", "nan", "none", "."}:
        key = "unknown"
    if key in VARIANT_TYPE_TO_INDEX:
        return VARIANT_TYPE_TO_INDEX[key]
    if "single" in key or "snv" in key or "snp" in key:
        return 1
    if "del" in key:
        return 2
    if "ins" in key:
        return 3
    if "indel" in key:
        return 4
    if "dup" in key:
        return 5
    return 0


def supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    strategy: str,
    tau: float,
) -> torch.Tensor:
    """Compute the supervised classification loss (optionally logit-adjusted)."""
    if strategy == "logit_adjustment":
        return logit_adjustment_cross_entropy(logits, targets, counts, tau=tau)
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def predict_proba(
    model: PAVRClassifier,
    features: dict[str, torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run batched inference and return class probabilities ``[N, C]``."""
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        idx = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
        vt = features.get("variant_type_index")
        out = model(
            features["h_ref"][idx].to(device).float(),
            features["h_alt"][idx].to(device).float(),
            features["variant_mask"][idx].to(device),
            features["padding_mask"][idx].to(device),
            variant_type=None if vt is None else vt[idx].to(device),
        )
        chunks.append(torch.softmax(out["prediction"], dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Compute macro-AUROC, macro-F1, accuracy, and balanced accuracy."""
    y_hat = proba.argmax(axis=1)
    return {
        "auroc_macro": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
    }


def train_one_seed(
    features: dict[str, torch.Tensor],
    y: np.ndarray,
    splits: np.ndarray,
    df: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Train PAVR for one seed and write checkpoint / predictions / metrics."""
    set_seed(seed)
    device = torch.device(args.device)
    model = PAVRClassifier(
        backbone_dim=int(features["h_ref"].shape[-1]),
        num_classes=len(CLASSES),
        common_dim=args.common_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        local_radius=args.local_radius,
        radius_temperature=args.radius_temperature,
        attention_excl_weight=args.attention_excl_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")
    # Prefer validation for early stopping; fall back to test if val is absent.
    monitor_idx = val_idx if len(val_idx) else test_idx

    counts = class_counts_from_labels(y[train_idx], num_classes=len(CLASSES))
    counts_t = torch.as_tensor(counts, dtype=torch.float32)

    if args.balanced_sampler:
        # Inverse-frequency sampling mitigates ClinVar class imbalance.
        sample_w = 1.0 / counts[y[train_idx].astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
        )
        loader = DataLoader(
            TensorDataset(torch.as_tensor(train_idx, dtype=torch.long)),
            batch_size=args.train_batch_size,
            sampler=sampler,
        )
    else:
        loader = DataLoader(
            TensorDataset(torch.as_tensor(train_idx, dtype=torch.long)),
            batch_size=args.train_batch_size,
            shuffle=True,
        )

    y_tensor = torch.as_tensor(y, dtype=torch.long)
    best_state = None
    best_score = -math.inf
    patience_left = args.patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for (idx,) in loader:
            vt = features.get("variant_type_index")
            out = model(
                features["h_ref"][idx].to(device).float(),
                features["h_alt"][idx].to(device).float(),
                features["variant_mask"][idx].to(device),
                features["padding_mask"][idx].to(device),
                variant_type=None if vt is None else vt[idx].to(device),
            )
            y_batch = y_tensor[idx].to(device)
            # Supervised CE + representation exclusivity from PAVRObjective.
            loss = supervised_loss(
                out["prediction"],
                y_batch,
                counts_t.to(device),
                strategy=args.imbalance_strategy,
                tau=args.logit_adjustment_tau,
            )
            loss = loss + out["loss_repr"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * idx.numel()
            total_n += int(idx.numel())

        mon_proba = predict_proba(model, features, monitor_idx, device, args.eval_batch_size)
        mon = metrics_from_proba(y[monitor_idx], mon_proba)
        score = mon["auroc_macro"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        row = {"epoch": epoch, "train_loss": total_loss / max(total_n, 1), **mon}
        history.append(row)
        print(f"[seed{seed}] epoch={epoch} loss={row['train_loss']:.4f} val_auroc={score:.4f}")
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_proba = predict_proba(model, features, test_idx, device, args.eval_batch_size)
    metrics = metrics_from_proba(y[test_idx], test_proba)

    out_dir = args.output_dir
    for sub in ("checkpoints", "histories", "predictions", "seed_metrics"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    tag = f"window_gastric_1k_seed{seed}"
    torch.save(best_state or model.state_dict(), out_dir / "checkpoints" / f"{tag}.pt")
    (out_dir / "histories" / f"{tag}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "seed_metrics" / f"{tag}.json").write_text(
        json.dumps({"task": "window_gastric_1k", "seed": seed, **metrics}, indent=2),
        encoding="utf-8",
    )

    pred_df = df.iloc[test_idx].copy()
    pred_df["y_true"] = y[test_idx]
    pred_df["y_pred"] = test_proba.argmax(axis=1)
    for i, name in enumerate(CLASSES):
        pred_df[f"prob_{name}"] = test_proba[:, i]
    cols = ["sample_id", "split", "label", "y_true", "y_pred"] + [f"prob_{c}" for c in CLASSES]
    pred_df[cols].to_csv(out_dir / "predictions" / f"{tag}.csv", index=False)
    return metrics


def main() -> None:
    """Load cached features, train across seeds, and write a summary JSON."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features: {args.features}")
    bundle = torch.load(args.features, map_location="cpu", weights_only=False)
    for key in ("h_ref", "h_alt", "variant_mask", "padding_mask"):
        if key not in bundle:
            raise KeyError(f"Missing key {key!r} in feature cache")

    df = pd.read_csv(args.csv)
    if len(df) != int(bundle["h_ref"].shape[0]):
        raise ValueError(f"CSV length {len(df)} != feature N {bundle['h_ref'].shape[0]}")

    if "label_id" in df.columns:
        y = df["label_id"].to_numpy(dtype=np.int64)
    else:
        label_to_id = {c: i for i, c in enumerate(CLASSES)}
        y = df["label"].map(label_to_id).to_numpy(dtype=np.int64)

    splits = df["split"].astype(str).to_numpy()
    # ClinVar demo CSV stores consequence under ``context``; accept ``variant_type`` too.
    type_col = "context" if "context" in df.columns else ("variant_type" if "variant_type" in df.columns else None)
    if type_col is not None:
        bundle["variant_type_index"] = torch.tensor(
            [variant_type_index(v) for v in df[type_col].tolist()],
            dtype=torch.long,
        )

    per_seed: list[dict[str, float]] = []
    for seed in args.seeds:
        m = train_one_seed(bundle, y, splits, df, seed, args)
        print(f"[seed{seed}] test AUROC={m['auroc_macro']:.4f} F1={m['f1_macro']:.4f}")
        per_seed.append({"seed": seed, **m})

    keys = ["auroc_macro", "f1_macro", "accuracy", "balanced_accuracy"]
    summary = {
        "method": "PAVR",
        "foundation_model": "Evo2_7B",
        "task": "window_gastric_1k",
        "features": str(args.features),
        "csv": str(args.csv),
        "seeds": args.seeds,
        "per_seed": per_seed,
        "metrics": {
            k: {
                "mean": float(np.mean([r[k] for r in per_seed])),
                "std": float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
            }
            for k in keys
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", args.output_dir / "summary.json")
    print(
        "mean±std AUROC="
        f"{summary['metrics']['auroc_macro']['mean']:.4f}±{summary['metrics']['auroc_macro']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
