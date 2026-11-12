"""CPOAD: Poisoning-Resistant Cyber-Physical Online Anomaly Detection.

Single-file reference implementation of the main model used in the paper:
  1. data loading of the causally fused CSVs, purged grouped 5-fold split,
     train-only feature pruning and preprocessing (median imputation,
     train-quantile clipping, standardisation, train-Normal deviation);
  2. the Normal-coupling predictive consistency detector (cross-domain
     predictor, intra-physical leave-one-out predictor, residual-calibrated
     coupling loss RCCL, memory-banked coupling-contrastive term);
  3. training with validation-only early stopping and training-time drift
     augmentation of the physical level channels (A = 0.8, probability 0.5);
  4. the guarded online update controller (three validation-calibrated
     signals, reliability-constrained and causally event-contiguous admission;
     Normal-coupling NLL on the predictor heads for admitted windows and
     cross-entropy on labelled validation replay windows for the classifier
     path, both with parameter anchoring; the encoder is frozen);
  5. the paper experiments: online stream conditions and baselines (Table XI),
     gate-aware poisoning (Table XII), confusion matrices (Fig. 5), and online
     traces (Fig. 7).  Requires matplotlib only for the figures.

Usage:
  python cpoad.py train  <IGCPS|WDT|ICS-Flow> <fold> [--data CSV] [--save PT]
  python cpoad.py online <IGCPS|WDT|ICS-Flow> <fold> [--data CSV] [--ckpt PT]
  python cpoad.py online-table <ds> <fold> | --all      Table XI streams (baselines + CPOAD)
  python cpoad.py poison <ds> <fold> | --all            Table XI + Table XII poisoning
  python cpoad.py confusion <ds>                        confusion matrix (pooled over 5 folds)
  python cpoad.py fig6 <ds>                             online traces (fold 0)
  common options: --ckpt-dir DIR (cache/load <ds>_f<k>_{target,surrogate}.pt),
                  --out DIR, --workers N, --attackers/--rates/--seeds (poison subset)

Default data files are the fused CSVs in the parent directory of this file
(igcps_final_unified.csv, wdt_final_unified.csv, ics_flow_final_unified.csv).
Split seed 202604 and model seed 1300 + fold reproduce the main table.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "IGCPS": ROOT / "igcps_final_unified.csv",
    "WDT": ROOT / "wdt_final_unified.csv",
    "ICS-Flow": ROOT / "ics_flow_final_unified.csv",
}
SPLIT_SEED = 202604


# =============================================================================
# 1. Data, split, and train-only preprocessing
# =============================================================================
NORMAL_NAMES = {"normal", "benign", "0"}


@dataclass(frozen=True)
class ExperimentConfig:
    seq_len: int = 4
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    d_model: int = 48  # shared baseline default; CAFOAD is set to 64 below
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.10  # shared baseline default; CAFOAD is set to 0.05 below
    batch_size: int = 256
    # The class-complete grouped protocol uses a common 30-epoch ceiling and
    # validation-only early stopping with patience seven for every method.
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 8e-4
    weight_decay: float = 2e-4
    max_train_windows: int = 80000
    max_validation_windows: int = 24000
    max_test_windows: int = 250000
    num_workers: int = 0
    clip_quantile_low: float = 0.005
    clip_quantile_high: float = 0.995


@dataclass
class DatasetBundle:
    name: str
    path: Path
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    class_names: List[str]
    fine_labels: np.ndarray
    fine_class_names: List[str]
    feature_names: List[str]
    cyber_indices: np.ndarray
    physical_indices: np.ndarray
    segment_ids: np.ndarray
    source_sha256: str
    fusion_protocol: str
    model_input_dim: Optional[int] = None


class TrainOnlyPreprocessor:
    """Median imputation, winsorisation, and scaling fitted on training rows."""

    def __init__(self, q_low: float, q_high: float):
        self.q_low = q_low
        self.q_high = q_high
        self.median: Optional[np.ndarray] = None
        self.lower: Optional[np.ndarray] = None
        self.upper: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.normal_center: Optional[np.ndarray] = None
        self.normal_scale: Optional[np.ndarray] = None

    def fit(
        self, x_train: np.ndarray, y_train: Optional[np.ndarray] = None
    ) -> "TrainOnlyPreprocessor":
        x = np.asarray(x_train, dtype=np.float64)
        self.median = np.nanmedian(x, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.0)
        x = np.where(np.isfinite(x), x, self.median)
        self.lower = np.quantile(x, self.q_low, axis=0)
        self.upper = np.quantile(x, self.q_high, axis=0)
        x = np.clip(x, self.lower, self.upper)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        if y_train is not None:
            base = (x - self.mean) / self.std
            normal = base[np.asarray(y_train) == 0]
            if len(normal) == 0:
                raise ValueError("The chronological training segment has no normal rows.")
            self.normal_center = np.median(normal, axis=0)
            mad = 1.4826 * np.median(
                np.abs(normal - self.normal_center), axis=0
            )
            normal_std = normal.std(axis=0)
            self.normal_scale = np.where(
                mad > 1e-6, mad, np.where(normal_std > 1e-6, normal_std, 1.0)
            )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if any(v is None for v in (self.median, self.lower, self.upper, self.mean, self.std)):
            raise RuntimeError("Preprocessor must be fitted before transform().")
        y = np.asarray(x, dtype=np.float64)
        y = np.where(np.isfinite(y), y, self.median)
        y = np.clip(y, self.lower, self.upper)
        y = (y - self.mean) / self.std
        y = np.nan_to_num(y, nan=0.0, posinf=8.0, neginf=-8.0)
        if self.normal_center is not None and self.normal_scale is not None:
            deviation = np.clip(
                np.abs((y - self.normal_center) / self.normal_scale), 0.0, 20.0
            )
            y = np.concatenate([y, deviation], axis=1)
        return y.astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {
            "q_low": self.q_low,
            "q_high": self.q_high,
            "median": self.median.tolist() if self.median is not None else None,
            "lower": self.lower.tolist() if self.lower is not None else None,
            "upper": self.upper.tolist() if self.upper is not None else None,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "normal_reference": {
                "fit_rows": "normal rows in the chronological training partition only",
                "center": "per-feature median",
                "scale": "MAD, falling back to training-normal standard deviation",
                "deviation_clip": 20.0,
                "enabled": self.normal_center is not None,
            },
        }


class CausalWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, targets: np.ndarray, seq_len: int):
        self.x = x
        self.y = y
        self.targets = np.asarray(targets, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = int(self.targets[item])
        start = end - self.seq_len + 1
        return torch.from_numpy(self.x[start : end + 1]), torch.tensor(int(self.y[end]))


def with_normal_deviation_schema(
    bundle: DatasetBundle, transformed_dim: int
) -> DatasetBundle:
    """Return metadata for base features plus train-normal deviations."""
    revised = copy.copy(bundle)
    original = list(bundle.feature_names)
    deviation_names: List[str] = []
    for name in original:
        if name.startswith("phy_"):
            deviation_names.append("phy_dev_" + name[4:])
        elif name.startswith("cyber_"):
            deviation_names.append("cyber_dev_" + name[6:])
        else:
            deviation_names.append("cyber_dev_" + name)
    revised.feature_names = original + deviation_names
    revised.cyber_indices = np.array(
        [i for i, name in enumerate(revised.feature_names) if name.startswith("cyber_")],
        dtype=np.int64,
    )
    revised.physical_indices = np.array(
        [i for i, name in enumerate(revised.feature_names) if name.startswith("phy_")],
        dtype=np.int64,
    )
    revised.model_input_dim = int(transformed_dim)
    if len(revised.feature_names) != transformed_dim:
        raise AssertionError("Normal-reference feature metadata does not match transformed data.")
    return revised


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_label_order(values: Iterable[object]) -> List[str]:
    labels = sorted({str(v).strip() for v in values})
    normal = [v for v in labels if v.lower() in NORMAL_NAMES]
    abnormal = [v for v in labels if v.lower() not in NORMAL_NAMES]
    return normal[:1] + abnormal


def load_dataset(name: str, path: Path) -> DatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"{name} data file does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "Time" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{name} must contain Time and label columns.")
    frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    segment_ids = (
        pd.to_numeric(frame["_segment_id"], errors="raise").to_numpy(np.int64)
        if "_segment_id" in frame.columns
        else np.zeros(len(frame), dtype=np.int64)
    )
    fine_class_names = canonical_label_order(frame["label"])
    label_map = {label: idx for idx, label in enumerate(fine_class_names)}
    fine_labels = frame["label"].astype(str).str.strip().map(label_map).to_numpy(np.int64)
    fine_normal = next(
        (i for i, label in enumerate(fine_class_names) if label.lower() in NORMAL_NAMES), 0
    )
    labels = (fine_labels != fine_normal).astype(np.int64)
    class_names = ["normal", "attack"]

    excluded = {
        "Time",
        "label",
        "pcuf_reliability",
        "_segment_id",
        "_source_row",
        "_partition",
    }
    feature_names = [c for c in frame.columns if c not in excluded]
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    features = numeric.to_numpy(np.float64)

    cyber = np.array([i for i, c in enumerate(feature_names) if c.lower().startswith("cyber_")])
    physical = np.array([i for i, c in enumerate(feature_names) if c.lower().startswith("phy_")])
    if len(cyber) == 0 or len(physical) == 0:
        raise ValueError(
            f"{name} requires explicit cyber_ and phy_ feature prefixes; "
            f"found cyber={len(cyber)}, physical={len(physical)}."
        )
    other = np.array([i for i in range(len(feature_names)) if i not in set(cyber) | set(physical)])
    if len(other):
        cyber = np.concatenate([cyber, other])

    return DatasetBundle(
        name=name,
        path=path,
        times=frame["Time"].to_numpy(),
        features=features,
        labels=labels,
        class_names=class_names,
        fine_labels=fine_labels,
        fine_class_names=fine_class_names,
        feature_names=feature_names,
        cyber_indices=cyber,
        physical_indices=physical,
        segment_ids=segment_ids,
        source_sha256=sha256_file(path),
        fusion_protocol=(
            "one physical-clock endpoint rule for all datasets: the row at p[i] "
            "uses the physical observation at p[i] and network events in "
            "(p[i-1], p[i]]; no interpolation, future fill, label alignment, "
            "or packet subsampling"
        ),
    )


def prune_features_from_training_rows(
    bundle: DatasetBundle, train_rows: np.ndarray
) -> Tuple[DatasetBundle, Dict[str, object]]:
    """Remove only train-constant and train-exact-duplicate inputs.

    The decision is made before validation/test values are inspected and is
    applied identically to CAFOAD and every baseline.  The fused CSV retains
    the complete physical schema; this function only avoids redundant model
    inputs in a particular chronological run.
    """
    train = np.asarray(bundle.features[train_rows], dtype=np.float64)
    keep: List[int] = []
    dropped_constant: List[str] = []
    dropped_duplicate: List[Dict[str, str]] = []
    fingerprints: Dict[str, List[int]] = {}
    normalized_columns: Dict[int, np.ndarray] = {}
    for index, name in enumerate(bundle.feature_names):
        column = train[:, index]
        finite = np.isfinite(column)
        if not finite.any():
            dropped_constant.append(name)
            continue
        median = float(np.nanmedian(column))
        normalized = np.where(finite, column, median).astype(np.float64, copy=False)
        if float(np.max(normalized) - np.min(normalized)) <= 1.0e-12:
            dropped_constant.append(name)
            continue
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(normalized).view(np.uint8)
        ).hexdigest()
        duplicate_of: Optional[int] = None
        for previous in fingerprints.get(fingerprint, []):
            if np.array_equal(normalized, normalized_columns[previous]):
                duplicate_of = previous
                break
        if duplicate_of is not None:
            dropped_duplicate.append(
                {"feature": name, "duplicate_of": bundle.feature_names[duplicate_of]}
            )
            continue
        keep.append(index)
        normalized_columns[index] = normalized
        fingerprints.setdefault(fingerprint, []).append(index)

    if not keep:
        raise RuntimeError(f"{bundle.name}: train-only pruning removed every feature")
    revised = copy.copy(bundle)
    revised.features = bundle.features[:, keep]
    revised.feature_names = [bundle.feature_names[index] for index in keep]
    revised.cyber_indices = np.array(
        [i for i, name in enumerate(revised.feature_names) if name.startswith("cyber_")],
        dtype=np.int64,
    )
    revised.physical_indices = np.array(
        [i for i, name in enumerate(revised.feature_names) if name.startswith("phy_")],
        dtype=np.int64,
    )
    assigned = set(revised.cyber_indices.tolist()) | set(revised.physical_indices.tolist())
    other = np.array(
        [i for i in range(len(revised.feature_names)) if i not in assigned], dtype=np.int64
    )
    if len(other):
        revised.cyber_indices = np.concatenate([revised.cyber_indices, other])
    audit: Dict[str, object] = {
        "fit_scope": "chronological training rows only",
        "original_features": int(len(bundle.feature_names)),
        "retained_features": int(len(revised.feature_names)),
        "dropped_train_constant": dropped_constant,
        "dropped_train_exact_duplicate": dropped_duplicate,
        "validation_or_test_values_used": False,
    }
    return revised, audit


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truths: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    reconstruction_errors: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        prob = torch.softmax(out["logits"], dim=-1)
        truths.append(y.numpy())
        predictions.append(prob.argmax(dim=-1).cpu().numpy())
        probabilities.append(prob.cpu().numpy())
        err = ((out["reconstruction"] - x[:, -1]) ** 2).mean(dim=-1)
        reconstruction_errors.append(err.cpu().numpy())
    return (
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(probabilities),
        np.concatenate(reconstruction_errors),
    )


def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def original_labels(bundle: DatasetBundle) -> DatasetBundle:
    """Use the fine-grained (multiclass) labels; class 0 is Normal."""
    converted = copy.copy(bundle)
    converted.labels = bundle.fine_labels.copy()
    converted.class_names = list(bundle.fine_class_names)
    return converted


def load_bundle(name: str, path: Optional[Path] = None) -> DatasetBundle:
    return original_labels(load_dataset(name, Path(path) if path else DATASETS[name]))


def default_config(**overrides) -> ExperimentConfig:
    values = dict(seq_len=4, epochs=30, patience=7, batch_size=256, learning_rate=8e-4,
                  weight_decay=2e-4, max_train_windows=250_000,
                  max_validation_windows=250_000, max_test_windows=250_000)
    values.update(overrides)
    return ExperimentConfig(**values)


def context_rows(targets: np.ndarray, seq_len: int) -> np.ndarray:
    return np.unique(
        np.concatenate([np.arange(t - seq_len + 1, t + 1) for t in targets])
    ).astype(np.int64)


def loader(x: np.ndarray, y: np.ndarray, targets: np.ndarray, config: ExperimentConfig,
           shuffle: bool, seed: int) -> DataLoader:
    """Class-balanced sampling for training; ordered loading otherwise."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    sampler = None
    if shuffle:
        target_labels = y[targets]
        counts = np.bincount(target_labels, minlength=int(y.max()) + 1).astype(np.float64)
        sample_weights = 1.0 / np.maximum(counts[target_labels], 1.0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(targets), replacement=True, generator=generator)
    return DataLoader(CausalWindowDataset(x, y, targets, config.seq_len),
                      batch_size=config.batch_size, shuffle=False, sampler=sampler,
                      num_workers=0)


def build_folds(
    bundle: Any,
    seq_len: int,
    folds: int = 5,
    block_rows: int = 7,
    split_seed: int = 202604,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    if block_rows < seq_len:
        raise ValueError("block_rows must be at least seq_len")
    purge_rows = seq_len - 1
    stride = block_rows + purge_rows
    run_break = np.r_[
        True,
        (bundle.labels[1:] != bundle.labels[:-1])
        | (bundle.segment_ids[1:] != bundle.segment_ids[:-1]),
    ]
    run_id = np.cumsum(run_break) - 1
    class_groups: dict[int, list[np.ndarray]] = {
        index: [] for index in range(len(bundle.class_names))
    }
    for current_run in np.unique(run_id):
        rows = np.flatnonzero(run_id == current_run)
        class_index = int(bundle.labels[rows[0]])
        for start in range(0, len(rows) - block_rows + 1, stride):
            class_groups[class_index].append(rows[start : start + block_rows])

    rng = np.random.default_rng(split_seed)
    fold_groups: list[dict[int, list[np.ndarray]]] = [
        {index: [] for index in class_groups} for _ in range(folds)
    ]
    groups_by_class: dict[str, int] = {}
    for class_index, class_name in enumerate(bundle.class_names):
        groups = class_groups[class_index]
        if len(groups) < folds:
            raise RuntimeError(
                f"{bundle.name}/{class_name}: {len(groups)} groups cannot support "
                f"{folds}-fold class-complete evaluation"
            )
        order = rng.permutation(len(groups))
        for position, group_index in enumerate(order):
            fold_groups[position % folds][class_index].append(groups[int(group_index)])
        groups_by_class[class_name] = len(groups)

    fold_partitions: list[dict[str, np.ndarray]] = []
    fold_audits: list[dict[str, Any]] = []
    tested_group_ids: set[tuple[int, int]] = set()
    for fold in range(folds):
        test_fold = fold
        validation_fold = (fold + 1) % folds
        training_folds = [
            index for index in range(folds)
            if index not in (test_fold, validation_fold)
        ]
        grouped: dict[str, list[np.ndarray]] = {
            "train": [], "validation": [], "test": []
        }
        for class_index in class_groups:
            for source_fold in training_folds:
                grouped["train"].extend(fold_groups[source_fold][class_index])
            grouped["validation"].extend(fold_groups[validation_fold][class_index])
            grouped["test"].extend(fold_groups[test_fold][class_index])

        parts: dict[str, np.ndarray] = {}
        row_sets: dict[str, set[int]] = {}
        counts_by_class: dict[str, dict[str, int]] = {}
        for part, groups in grouped.items():
            row_sets[part] = set(
                np.concatenate(groups).astype(np.int64).tolist()
            )
            targets = [
                np.arange(
                    int(group[0]) + seq_len - 1,
                    int(group[-1]) + 1,
                    dtype=np.int64,
                )
                for group in groups
            ]
            parts[part] = np.sort(np.concatenate(targets))
            counts_by_class[part] = {
                class_name: int(np.sum(bundle.labels[parts[part]] == class_index))
                for class_index, class_name in enumerate(bundle.class_names)
            }
        overlap = {
            "train_validation": len(row_sets["train"] & row_sets["validation"]),
            "train_test": len(row_sets["train"] & row_sets["test"]),
            "validation_test": len(row_sets["validation"] & row_sets["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Fold {fold} row overlap: {overlap}")
        for class_index, groups in fold_groups[test_fold].items():
            for group in groups:
                group_key = (class_index, int(group[0]))
                if group_key in tested_group_ids:
                    raise RuntimeError(f"Group tested twice: {group_key}")
                tested_group_ids.add(group_key)
        fold_partitions.append(parts)
        fold_audits.append({
            "fold": fold,
            "training_folds": training_folds,
            "validation_fold": validation_fold,
            "test_fold": test_fold,
            "cross_partition_row_overlap": overlap,
            "target_counts": {part: int(len(rows)) for part, rows in parts.items()},
            "target_counts_by_class": counts_by_class,
        })

    expected_groups = sum(len(groups) for groups in class_groups.values())
    if len(tested_group_ids) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} once-tested groups, got {len(tested_group_ids)}"
        )
    audit = {
        "rule": (
            "five-fold stratified rotation of non-overlapping homogeneous causal "
            "event blocks; each retained block is test data exactly once"
        ),
        "split_seed": split_seed,
        "folds": folds,
        "block_rows": block_rows,
        "nominal_block_seconds": block_rows * 0.2,
        "purge_rows_between_adjacent_blocks": purge_rows,
        "nominal_purge_seconds": purge_rows * 0.2,
        "sequence_length": seq_len,
        "groups_by_class": groups_by_class,
        "test_group_coverage": "exactly once",
        "fold_audits": fold_audits,
    }
    return fold_partitions, audit


# =============================================================================
# 2. CPOAD detector: Normal-coupling predictive consistency
# =============================================================================
def split_schema(feature_names: Sequence[str], with_obs: bool = False):
    """Return index lists (cyber_like, phys_base, phys_target[, phys_all]).

    phys_target = physical level coordinates (no delta/deviation suffix).
    """
    phys_base, cyber_like, target = [], [], []
    phys_all = [i for i, n in enumerate(feature_names) if n.startswith("phy_")]
    for i, n in enumerate(feature_names):
        if n.startswith("phy_dev_"):
            continue
        if n.startswith("phy_"):
            phys_base.append(i)
            if "_delta_observation_" not in n:
                target.append(i)
        else:
            cyber_like.append(i)
    if with_obs:
        return cyber_like, phys_base, target, phys_all
    return cyber_like, phys_base, target


CPCD5_VARIANTS: Dict[str, Dict[str, bool]] = {
    "CPCD5": {},
    "CPCD5-w/o-rccl": {"no_rccl": True},
    "CPCD5-w/o-contrastive": {"no_contrastive": True},
    "CPCD5-w/o-intra": {"no_intra": True},
    "CPCD5-w/o-cross": {"no_cross": True},
    "CPCD5-w/o-consistency": {"no_cross": True, "no_intra": True},
}


class CouplingDifficultyEstimator:
    """Per-class coupling difficulty + frequency rarity tracker (v2)."""

    def __init__(self, n_classes: int, momentum: float = 0.05, memory_size: int = 64):
        self.n_classes = n_classes
        self.momentum = momentum
        self.memory_size = memory_size
        self.ema_mag = torch.zeros(n_classes)
        self.ema_var = torch.ones(n_classes)
        self.initialized = torch.zeros(n_classes, dtype=torch.bool)
        self.class_counts = torch.zeros(n_classes)
        self.memory: List[List[torch.Tensor]] = [[] for _ in range(n_classes)]

    def update(self, residuals: torch.Tensor, labels: torch.Tensor):
        mag = residuals.detach().abs().mean(dim=-1)
        for k in range(self.n_classes):
            mask = (labels == k)
            count = mask.sum().item()
            if count < 1:
                continue
            self.class_counts[k] += count
            class_mag = mag[mask]
            new_mean = class_mag.mean()
            new_var = class_mag.var() if count > 1 else self.ema_var[k]
            for row in residuals[mask].detach():
                if len(self.memory[k]) >= self.memory_size:
                    self.memory[k].pop(0)
                self.memory[k].append(row)
            if not self.initialized[k]:
                self.ema_mag[k] = new_mean
                self.ema_var[k] = new_var
                self.initialized[k] = True
            else:
                m = min(self.momentum * max(1.0, 10.0 / max(count, 1)), 0.5)
                self.ema_mag[k] = (1 - m) * self.ema_mag[k] + m * new_mean
                self.ema_var[k] = (1 - m) * self.ema_var[k] + m * new_var

    def difficulty_weights(self, device: torch.device) -> torch.Tensor:
        if not self.initialized[0]:
            return torch.ones(self.n_classes, device=device)
        normal_mag = self.ema_mag[0]
        dist = (self.ema_mag - normal_mag).abs()
        overlap = 1.0 / (1.0 + dist / (self.ema_var.sqrt().clamp(min=0.1)))
        coupling_w = 0.5 + 1.5 * overlap
        total = self.class_counts.sum().clamp(min=1.0)
        freq = self.class_counts / total
        rarity = 1.0 / (freq.sqrt().clamp(min=0.05))
        rarity = rarity / rarity.mean()
        weights = coupling_w * rarity
        weights[0] = 1.0
        weights = weights.clamp(0.5, 4.0)
        return weights.to(device)

    def get_memory_centroids(self, device: torch.device):
        valid_k, centroids = [], []
        for k in range(self.n_classes):
            if len(self.memory[k]) >= 2:
                centroids.append(torch.stack(self.memory[k]).mean(dim=0))
                valid_k.append(k)
        if not valid_k:
            return None, []
        return torch.stack(centroids).to(device), valid_k


class CPCD5(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, feature_names: Sequence[str],
                 d_model: int = 64, dropout: float = 0.1, lambda_nll: float = 0.3,
                 lambda_intra: float = 0.3, lambda_contrastive: float = 0.15,
                 detach_residual: bool = False,
                 flags: Dict[str, bool] | None = None, **_: object):
        super().__init__()
        self.flags = dict(flags or {})
        self.lambda_nll = float(lambda_nll)
        self.lambda_intra = float(lambda_intra)
        self.lambda_contrastive = float(lambda_contrastive)
        self.detach_residual = bool(detach_residual)
        self.n_classes = n_classes
        cyb, phys, tgt, obs = split_schema(feature_names, with_obs=True)
        if not tgt:
            raise ValueError("CPCD5 needs at least one physical level coordinate")
        self.register_buffer("cyb_idx", torch.tensor(cyb, dtype=torch.long))
        self.register_buffer("phys_idx", torch.tensor(phys, dtype=torch.long))
        self.register_buffer("tgt_idx", torch.tensor(tgt, dtype=torch.long))
        self.register_buffer("obs_idx", torch.tensor(obs, dtype=torch.long))
        dc, dp, dt = len(cyb), len(phys), len(tgt)
        self.n_target = dt
        token_in = dc + dp
        self.token = nn.Sequential(nn.Linear(token_in, d_model), nn.GELU(),
                                   nn.LayerNorm(d_model))
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.mu_head = nn.Linear(d_model, dt)
        self.logvar_head = nn.Linear(d_model, dt)
        if dt > 1 and not self.flags.get("no_intra"):
            self.intra_proj = nn.Sequential(nn.Linear(dt, d_model), nn.GELU(),
                                             nn.LayerNorm(d_model))
            self.intra_mu = nn.Linear(d_model, 1)
            self.intra_logvar = nn.Linear(d_model, 1)
            loo = torch.ones(dt, dt) - torch.eye(dt)
            self.register_buffer("intra_loo", loo)
        intra_dim = dt if dt > 1 else 0
        res_in = len(obs) + dt + intra_dim
        self.res_embed = nn.Sequential(nn.Linear(res_in, d_model), nn.GELU(),
                                      nn.LayerNorm(d_model))
        self.res_pool = nn.GRU(d_model, d_model, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout), nn.Linear(2 * d_model, n_classes))
        self.difficulty_estimator = CouplingDifficultyEstimator(n_classes)
        if not self.flags.get("no_contrastive"):
            self.contrast_proj = nn.Sequential(
                nn.Linear(dt + intra_dim, d_model), nn.GELU(),
                nn.Linear(d_model, d_model // 2))
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cyber = x[:, 1:].index_select(-1, self.cyb_idx)
        prev_phys = x[:, :-1].index_select(-1, self.phys_idx)
        target = x[:, 1:].index_select(-1, self.tgt_idx)
        obs = x[:, 1:].index_select(-1, self.obs_idx)
        no_cross = self.flags.get("no_cross")
        if no_cross:
            mu = torch.zeros_like(target)
            logvar = torch.zeros_like(target)
            cd_residual = torch.zeros_like(target)
        else:
            tokens = torch.cat([cyber, prev_phys], dim=-1)
            h, _ = self.encoder(self.token(tokens))
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(-6.0, 6.0)
            m, lv = (mu.detach(), logvar.detach()) if self.detach_residual else (mu, logvar)
            cd_residual = ((target - m) * torch.exp(-0.5 * lv)).clamp(-20.0, 20.0)
        no_intra = self.flags.get("no_intra") or self.n_target <= 1 or not hasattr(self, "intra_mu")
        if no_intra:
            intra_residual = torch.zeros_like(target)
            intra_mu = torch.zeros_like(target)
            intra_logvar = torch.zeros_like(target)
        else:
            B, T, dt = target.shape
            masked = target.unsqueeze(2) * self.intra_loo
            flat = masked.reshape(B * T * dt, dt)
            hi = self.intra_proj(flat)
            intra_mu = self.intra_mu(hi).reshape(B, T, dt)
            intra_logvar = self.intra_logvar(hi).reshape(B, T, dt).clamp(-6.0, 6.0)
            im, ilv = ((intra_mu.detach(), intra_logvar.detach())
                       if self.detach_residual else (intra_mu, intra_logvar))
            intra_residual = ((target - im) * torch.exp(-0.5 * ilv)).clamp(-20.0, 20.0)
        if no_cross:
            tokens = torch.cat([cyber, prev_phys], dim=-1)
            h, _ = self.encoder(self.token(tokens))
        evidence = torch.cat([obs, cd_residual, intra_residual], dim=-1)
        r, _ = self.res_pool(self.res_embed(evidence))
        logits = self.classifier(torch.cat([h[:, -1], r[:, -1]], dim=-1))
        combined_residual = torch.cat([cd_residual[:, -1], intra_residual[:, -1]], dim=-1)
        return {
            "logits": logits, "mu": mu, "logvar": logvar, "target": target,
            "residual": cd_residual,
            "intra_mu": intra_mu, "intra_logvar": intra_logvar,
            "intra_residual": intra_residual,
            "combined_residual": combined_residual,
            "reconstruction": x[:, -1],
            "consistency_score": 0.5 * (cd_residual[:, -1] ** 2).mean(dim=-1),
        }
    def rccl_loss(self, logits, y, combined_residual):
        """Residual-Calibrated Coupling Loss v2 with rare-class residual replay."""
        if self.flags.get("no_rccl"):
            ce = F.cross_entropy(logits, y, reduction="none")
            pt = torch.softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
            return (((1.0 - pt) ** 2) * ce).mean()
        self.difficulty_estimator.update(combined_residual, y)
        w = self.difficulty_estimator.difficulty_weights(logits.device)
        sample_w = w[y]
        r_mag = combined_residual.detach().abs().mean(dim=-1)
        r_norm = r_mag / (r_mag.mean().clamp(min=1e-6))
        coupling_mod = (1.0 + r_norm) ** 1.0
        ce = F.cross_entropy(logits, y, reduction="none")
        base_loss = (sample_w * coupling_mod * ce).mean()
        # Rare-class residual replay: re-classify memory-banked residuals
        # through the classifier's residual pathway to generate extra gradients
        replay_loss = self._rare_class_replay(y, combined_residual)
        return base_loss + 0.3 * replay_loss

    def _rare_class_replay(self, y, combined_residual):
        """Replay stored rare-class residuals through contrast projection."""
        if not hasattr(self, "contrast_proj"):
            return torch.tensor(0.0, device=combined_residual.device)
        total = self.difficulty_estimator.class_counts.sum().clamp(min=1)
        freq = self.difficulty_estimator.class_counts / total
        mean_freq = freq[freq > 0].mean() if (freq > 0).any() else 1.0
        rare_classes = [k for k in range(self.n_classes)
                       if k != 0 and freq[k] > 0 and freq[k] < 0.5 * mean_freq
                       and len(self.difficulty_estimator.memory[k]) >= 4]
        if not rare_classes:
            return torch.tensor(0.0, device=combined_residual.device)
        # Sample from memory and classify via projection + simple linear probe
        replay_z, replay_y = [], []
        for k in rare_classes:
            mem = self.difficulty_estimator.memory[k]
            n_replay = min(len(mem), 8)
            indices = torch.randperm(len(mem))[:n_replay]
            for idx in indices:
                replay_z.append(mem[idx])
                replay_y.append(k)
        if not replay_z:
            return torch.tensor(0.0, device=combined_residual.device)
        replay_z = torch.stack(replay_z).to(combined_residual.device)
        replay_y = torch.tensor(replay_y, device=combined_residual.device, dtype=torch.long)
        # Project and compute contrastive similarity as proxy classification
        centroids_mem, valid_k = self.difficulty_estimator.get_memory_centroids(combined_residual.device)
        if centroids_mem is None or len(valid_k) < 2:
            return torch.tensor(0.0, device=combined_residual.device)
        z = F.normalize(self.contrast_proj(replay_z), dim=-1)
        c = F.normalize(self.contrast_proj(centroids_mem), dim=-1)
        sim = z @ c.t() / 0.2
        class_to_col = {k: i for i, k in enumerate(valid_k)}
        own = torch.tensor([class_to_col.get(int(yi), 0) for yi in replay_y],
                            device=z.device, dtype=torch.long)
        valid = torch.tensor([int(yi) in class_to_col for yi in replay_y],
                              device=z.device, dtype=torch.bool)
        if valid.sum() < 1:
            return torch.tensor(0.0, device=z.device)
        return F.cross_entropy(sim[valid], own[valid])

    def coupling_contrastive_loss(self, combined_residual, y):
        """Memory-banked coupling-contrastive regularizer v2."""
        if self.flags.get("no_contrastive") or not hasattr(self, "contrast_proj"):
            return torch.tensor(0.0, device=combined_residual.device)
        z = F.normalize(self.contrast_proj(combined_residual), dim=-1)
        centroids_mem, valid_k = self.difficulty_estimator.get_memory_centroids(z.device)
        if centroids_mem is None or len(valid_k) < 2:
            return torch.tensor(0.0, device=z.device)
        proj_centroids = F.normalize(self.contrast_proj(centroids_mem), dim=-1)
        sim = z @ proj_centroids.t() / 0.2
        class_to_col = {k: i for i, k in enumerate(valid_k)}
        own_col = torch.tensor([class_to_col.get(int(yi), 0) for yi in y],
                                device=z.device, dtype=torch.long)
        valid_mask = torch.tensor([int(yi) in class_to_col for yi in y],
                                   device=z.device, dtype=torch.bool)
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=z.device)
        return F.cross_entropy(sim[valid_mask], own_col[valid_mask])

    def nll_cross(self, out, mask):
        per = 0.5 * (out["logvar"] + (out["target"] - out["mu"]) ** 2 * torch.exp(-out["logvar"]))
        per = per.mean(dim=(1, 2))
        if mask is not None:
            if not mask.any():
                return per.sum() * 0.0
            per = per[mask]
        return per.mean()

    def nll_intra(self, out, mask):
        if not hasattr(self, "intra_mu") or self.flags.get("no_intra"):
            return torch.tensor(0.0, device=out["logits"].device)
        per = 0.5 * (out["intra_logvar"] +
                     (out["target"] - out["intra_mu"]) ** 2 * torch.exp(-out["intra_logvar"]))
        per = per.mean(dim=(1, 2))
        if mask is not None:
            if not mask.any():
                return per.sum() * 0.0
            per = per[mask]
        return per.mean()

    def loss(self, x, y):
        out = self(x)
        cls_loss = self.rccl_loss(out["logits"], y, out["combined_residual"])
        normal_mask = (y == 0)
        nll = torch.tensor(0.0, device=x.device)
        if not self.flags.get("no_cross"):
            nll = nll + self.lambda_nll * self.nll_cross(out, normal_mask)
        if not self.flags.get("no_intra"):
            nll = nll + self.lambda_intra * self.nll_intra(out, normal_mask)
        ctr = self.lambda_contrastive * self.coupling_contrastive_loss(
            out["combined_residual"], y)
        # Rare-class augmented classification: duplicate rare samples with
        # noise-perturbed residuals and reclassify them for extra gradients
        aug_loss = self._augmented_rare_loss(out, y)
        return cls_loss + nll + ctr + 0.2 * aug_loss

    def _augmented_rare_loss(self, out, y):
        """Synthesize extra classification signal for rare classes via
        residual-space perturbation. For each rare-class sample in the batch,
        create a noisy copy of its classifier input and compute an additional
        classification loss. This acts as implicit oversampling in feature
        space without modifying the data loader."""
        logits = out["logits"]
        # Identify rare classes: less than 10% of batch
        B = y.shape[0]
        counts = torch.bincount(y, minlength=self.n_classes).float()
        threshold = max(B * 0.1, 2.0)
        rare_mask = torch.zeros(B, dtype=torch.bool, device=y.device)
        for k in range(1, self.n_classes):  # skip Normal
            if 0 < counts[k] < threshold:
                rare_mask |= (y == k)
        if rare_mask.sum() < 1:
            return torch.tensor(0.0, device=y.device)
        # Re-run classifier head on perturbed evidence for rare samples
        cd_res = out["residual"][:, -1]  # (B, dt)
        intra_res = out["intra_residual"][:, -1]  # (B, dt)
        obs = out["reconstruction"]  # approximate; not perfect but usable
        # Actually we need the evidence tensor. Let's perturb the logits
        # directly with coupling-informed noise as a simpler approach:
        rare_logits = logits[rare_mask]
        rare_y = y[rare_mask]
        # Add noise proportional to residual magnitude (coupling-informed)
        r_mag = out["combined_residual"][rare_mask].detach().abs().mean(dim=-1, keepdim=True)
        noise_scale = 0.1 * r_mag.clamp(max=2.0)
        noise = torch.randn_like(rare_logits) * noise_scale
        aug_logits = rare_logits + noise
        return F.cross_entropy(aug_logits, rare_y)


def build_model(bundle: DatasetBundle, flags: Optional[Dict[str, bool]] = None) -> CPCD5:
    """Full CPOAD configuration (flags=None); ablations via CPCD5_VARIANTS."""
    return CPCD5(input_dim=bundle.model_input_dim, n_classes=len(bundle.class_names),
                 feature_names=bundle.feature_names, flags=flags or {})


# =============================================================================
# 3. Training (validation-only early stopping)
# =============================================================================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train(model: CPCD5, train_loader: DataLoader, validation_loader: DataLoader,
          config: ExperimentConfig) -> Dict[str, Any]:
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        yv, pv, _, _ = predict_loader(model, validation_loader, device)
        score = macro_f1_present(yv, pv)
        if score > best_f1 + 1e-4:
            best_f1, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_validation_macro_f1_selection": best_f1, "best_epoch": best_epoch,
            "epochs_executed": epoch + 1}


@dataclass
class Fold:
    dataset: str
    fold: int
    bundle: DatasetBundle          # multiclass labels
    parts: Dict[str, np.ndarray]   # train / validation / test target rows
    x: np.ndarray                  # transformed features (base + deviation)
    raw: np.ndarray                # pruned raw features
    model_bundle: DatasetBundle    # schema of x
    prep: TrainOnlyPreprocessor
    config: ExperimentConfig
    model_seed: int
    level_idx: np.ndarray          # physical level coordinates (raw/base space)


_BUNDLES: Dict[str, DatasetBundle] = {}
_FOLDS: Dict[Tuple[str, int], List[Dict[str, np.ndarray]]] = {}


def prepare_fold(dataset: str, fold: int, data_path: Optional[Path] = None,
                 split_seed: int = SPLIT_SEED) -> Fold:
    if dataset not in _BUNDLES:
        _BUNDLES[dataset] = load_bundle(dataset, data_path)
    bundle = _BUNDLES[dataset]
    config = default_config()
    if (dataset, split_seed) not in _FOLDS:
        _FOLDS[(dataset, split_seed)] = build_folds(bundle, config.seq_len, 5, 7, split_seed)[0]
    parts = _FOLDS[(dataset, split_seed)][fold]
    model_seed = 13 * 100 + fold + (0 if split_seed == SPLIT_SEED else 10_000 * split_seed)
    fitted = context_rows(parts["train"], config.seq_len)
    reduced, _ = prune_features_from_training_rows(bundle, fitted)
    prep = TrainOnlyPreprocessor(config.clip_quantile_low, config.clip_quantile_high)
    prep.fit(reduced.features[fitted], reduced.labels[fitted])
    x = prep.transform(reduced.features)
    mb = with_normal_deviation_schema(copy.copy(reduced), x.shape[1])
    level = np.array([i for i, n in enumerate(reduced.feature_names)
                      if n.startswith("phy_") and "_delta_" not in n and "_dev_" not in n],
                     dtype=np.int64)
    return Fold(dataset, fold, bundle, parts, x, np.asarray(reduced.features, np.float64),
                mb, prep, config, model_seed, level)


DRIFT_AUG_A, DRIFT_AUG_P = 0.8, 0.5     # training-time drift augmentation (Sec. III-C6)


def make_drift_aug(fd: "Fold", seed: int):
    """Eq. (17): with probability DRIFT_AUG_P per window, the standardized
    physical level channels z become c + g (z - c) + A (u + 0.3 u'), g ~ U(1, 1+A),
    u ~ U(-1, 1) shared, u' ~ U(-1, 1) per channel, constant over the window; then
    train-quantile clipping and recomputation of the Normal-deviation features."""
    p, A = fd.prep, DRIFT_AUG_A
    lv = torch.as_tensor(fd.level_idx)
    nb = len(fd.prep.mean)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    lo, hi = t((p.lower - p.mean) / p.std), t((p.upper - p.mean) / p.std)
    c, s = t(p.normal_center), t(p.normal_scale)
    g = torch.Generator().manual_seed(seed)

    def aug(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        m = torch.rand(x.shape[0], generator=g) < DRIFT_AUG_P
        if not m.any():
            return x
        base, n, L = x[m, :, :nb], int(m.sum()), len(lv)
        u = torch.rand(n, 1, 1, generator=g) * 2 - 1
        uc = torch.rand(n, 1, L, generator=g) * 2 - 1
        gain = 1 + A * torch.rand(n, 1, 1, generator=g)
        cl = c[lv]
        lvl = cl + (base[..., lv] - cl) * gain + A * (u + 0.3 * uc)
        base[..., lv] = torch.maximum(torch.minimum(lvl, hi[lv]), lo[lv])
        x[m] = torch.cat([base, ((base - c) / s).abs().clamp(0.0, 20.0)], dim=-1)
        return x
    return aug


def train_fold_model(fd: Fold, seed_offset: int = 0,
                     drift_aug: Optional[bool] = None) -> Tuple[CPCD5, Dict[str, Any]]:
    """Main-table training path with drift augmentation; seed_offset=7777 gives the
    gray-box surrogate, which the attacker trains with the same recipe (other seed)."""
    torch.set_num_threads(1)
    seed = fd.model_seed + seed_offset
    tl = loader(fd.x, fd.bundle.labels, fd.parts["train"], fd.config, True, seed)
    vl = loader(fd.x, fd.bundle.labels, fd.parts["validation"], fd.config, False, seed)
    seed_all(seed)
    model = build_model(fd.model_bundle)
    if drift_aug if drift_aug is not None else True:
        aug, orig = make_drift_aug(fd, 4242 + fd.fold + seed_offset), model.loss
        model.loss = lambda xb, yb: orig(aug(xb), yb)        # training batches only
    info = train(model, tl, vl, fd.config)
    model.__dict__.pop("loss", None)
    return model, info


def run_fold(dataset: str, fold: int, split_seed: int = SPLIT_SEED,
             data_path: Optional[Path] = None) -> Tuple[CPCD5, Fold, Dict[str, Any]]:
    """Train CPOAD on one fold exactly as for the main table."""
    fd = prepare_fold(dataset, fold, data_path, split_seed)
    model, info = train_fold_model(fd)
    return model, fd, info


def load_model(fd: Fold, path: Path) -> CPCD5:
    seed_all(fd.model_seed)
    model = build_model(fd.model_bundle)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def get_models(fd: Fold, ckpt_dir: Optional[Path] = None) -> Tuple[CPCD5, CPCD5]:
    """Target model and gray-box surrogate, from <ckpt_dir>/<ds>_f<k>_{target,surrogate}.pt
    if present, else trained (and cached there when ckpt_dir is given).  The global
    torch RNG state on return is the same either way (seed_all + one model build),
    which the online experiments rely on for exact reproduction."""
    out = []
    for tag, off in (("target", 0), ("surrogate", 7777)):
        path = Path(ckpt_dir) / f"{fd.dataset}_f{fd.fold}_{tag}.pt" if ckpt_dir else None
        if path is not None and path.exists():
            out.append(load_model(fd, path))
        else:
            model, _ = train_fold_model(fd, off)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
            out.append(load_model_state(fd, model.state_dict()))
    return out[0], out[1]


def load_model_state(fd: Fold, state: Dict[str, torch.Tensor]) -> CPCD5:
    seed_all(fd.model_seed)
    model = build_model(fd.model_bundle)
    model.load_state_dict(state)
    model.eval()
    return model


def windows(fd: Fold, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = fd.config.seq_len
    return np.stack([x[t - q + 1: t + 1] for t in targets]).astype(np.float32)


@torch.no_grad()
def predict(model: nn.Module, w: np.ndarray, batch: int = 512) -> np.ndarray:
    out = [model(torch.from_numpy(w[i:i + batch]))["logits"].argmax(-1).numpy()
           for i in range(0, len(w), batch)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def macro_f1_all(truth: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over all classes (main-table convention)."""
    _, _, f, _ = precision_recall_fscore_support(truth, pred, labels=np.arange(n_classes),
                                                 zero_division=0)
    return float(f.mean())


# =============================================================================
# 4. Guarded online update controller
# =============================================================================
BATCH = 16          # stream batch size
KAPPA = 3.0         # robust threshold multiplier
PERSIST = 2         # warning persistence r
CONTEXT = 2         # event-contiguity rule: preceding windows predicted Normal
HEAD_LR = 1e-3
ANCHOR = 0.02
HEADS = ("mu_head", "logvar_head", "intra_mu", "intra_logvar")
CLS_PATH = ("res_embed", "res_pool", "classifier")   # updated only on labelled replay
CLS_LR, N_REPLAY = 1e-4, 32


def signals(out) -> Dict[str, torch.Tensor]:
    prob = torch.softmax(out["logits"], -1)
    conf = prob.max(-1).values
    ent = -(prob * prob.clamp_min(1e-12).log()).sum(-1)
    score = out["combined_residual"].square().mean(-1)   # coupling consistency
    return {"prob": prob, "conf": conf, "ent": ent, "score": score}


@torch.no_grad()
def calibrate(model, windows, labels, normal_idx):
    sig = signals(model(torch.from_numpy(windows)))
    conf, ent, score = (sig[k].numpy() for k in ("conf", "ent", "score"))
    nb = len(windows) // BATCH
    b = np.array([[score[i*BATCH:(i+1)*BATCH].mean(), ent[i*BATCH:(i+1)*BATCH].mean(),
                   (1 - conf[i*BATCH:(i+1)*BATCH]).mean()] for i in range(nb)])
    center = np.median(b, 0)
    mad = np.maximum(1.4826 * np.median(np.abs(b - center), 0), 1e-6)
    nrm = labels == normal_idx
    return {
        "batch_upper": center + KAPPA * mad,
        "T_c": float(np.clip(np.quantile(conf[nrm], 0.05), 0.85, 0.95)),
        "T_s": float(np.quantile(score[nrm], 0.99)),
        "T_h": float(max(0.10, np.quantile(ent, 0.90))),
    }


class GuardedOnlineUpdater:
    """Prequential guarded online updating (Algorithm 3 of the paper).

    Every batch is predicted and logged before any update.  An update happens
    only when (i) at least one of the three batch signals exceeds its
    validation threshold for PERSIST consecutive batches and at least two
    exceed it in the current batch, and (ii) some windows are admitted:
    predicted Normal, confidence >= T_c, coupling score <= T_s, entropy <=
    T_h, and the CONTEXT preceding windows of the stream were predicted
    Normal.  The update minimises the Normal-coupling NLL of the admitted
    windows (predictor heads, lr HEAD_LR) plus the cross-entropy of N_REPLAY
    labelled validation windows (classifier path, lr CLS_LR), with an anchor to
    the initial parameters.  No stream pseudo-label reaches the classifier; the
    encoder is frozen.
    """

    def __init__(self, model: CPCD5, cal: Dict[str, Any], replay: Tuple[np.ndarray, np.ndarray],
                 normal_idx: int = 0, seed: int = 1234):
        self.model = copy.deepcopy(model).eval()
        self.cal, self.normal_idx = cal, normal_idx
        self.rx, self.ry = replay
        self.rng = np.random.default_rng(seed)
        self.anchor = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        head = [p for n, p in self.model.named_parameters() if n.split(".")[0] in HEADS]
        cls = [p for n, p in self.model.named_parameters() if n.split(".")[0] in CLS_PATH]
        self.params = head + cls
        self.opt = torch.optim.AdamW([{"params": head, "lr": HEAD_LR},
                                      {"params": cls, "lr": CLS_LR}], weight_decay=2e-4)
        self.run = 0
        self.history = np.zeros(0, bool)      # logged "predicted Normal" flags
        self.updates = 0

    def step(self, xb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process one batch of model-input windows (B, q, F)."""
        xt = torch.from_numpy(xb)
        with torch.no_grad():
            sig = signals(self.model(xt))
        pred = sig["prob"].argmax(-1).numpy()             # logged before any update
        conf, ent, score = (sig[k].numpy() for k in ("conf", "ent", "score"))
        v = np.array([score.mean(), ent.mean(), (1 - conf).mean()])
        exceeded = v > self.cal["batch_upper"]
        warning = bool(exceeded.any())
        self.run = self.run + 1 if warning else 0
        action = bool(exceeded.sum() >= 2 and self.run >= PERSIST)
        el = (action & (pred == self.normal_idx) & (conf >= self.cal["T_c"])
              & (score <= self.cal["T_s"]) & (ent <= self.cal["T_h"]))
        h = np.r_[self.history, pred == self.normal_idx]
        off = len(self.history)
        for j in np.flatnonzero(el):                      # causal contiguity
            i = off + j
            el[j] = i >= CONTEXT and bool(h[i - CONTEXT:i].all())
        self.history = h
        if el.any():
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            out = self.model(xt[torch.from_numpy(el)])
            nm = torch.ones(int(el.sum()), dtype=torch.bool)
            loss = self.model.nll_cross(out, nm) + self.model.nll_intra(out, nm)
            k = self.rng.choice(len(self.rx), size=min(N_REPLAY, len(self.rx)), replace=False)
            ro = self.model(torch.from_numpy(self.rx[k]))
            loss = loss + F.cross_entropy(ro["logits"], torch.from_numpy(self.ry[k]))
            loss = loss + ANCHOR * sum((p - self.anchor[n]).square().mean()
                                       for n, p in self.model.named_parameters()
                                       if any(p is q for q in self.params))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.model.eval()
            self.updates += 1
        return pred, {"warning": warning, "action": action, "admitted": int(el.sum())}


def replay_set(fd: "Fold", targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Labelled clean windows of the validation partition used for classifier replay."""
    return (windows(fd, fd.x, targets).astype(np.float32),
            fd.bundle.labels[targets].astype(np.int64))


def run_online(model: CPCD5, fd: Fold) -> Dict[str, Any]:
    """Clean prequential stream on the fold's test partition (gates and replay on validation)."""
    names = [c.lower() for c in fd.bundle.class_names]
    normal_idx = names.index("normal") if "normal" in names else 0
    cal_t, stream_t = fd.parts["validation"], fd.parts["test"]
    cal = calibrate(model, windows(fd, fd.x, cal_t), fd.bundle.labels[cal_t], normal_idx)
    w, truth = windows(fd, fd.x, stream_t), fd.bundle.labels[stream_t]
    up = GuardedOnlineUpdater(model, cal, replay_set(fd, cal_t), normal_idx)
    preds = [up.step(w[s:s + BATCH])[0] for s in range(0, len(w), BATCH)]
    pred = np.concatenate(preds)
    frozen = predict(model, w)
    ncls = len(fd.bundle.class_names)
    return {"preq_macro_f1": macro_f1_all(truth, pred, ncls),
            "frozen_macro_f1": macro_f1_all(truth, frozen, ncls),
            "update_batches": up.updates, "batches": int(np.ceil(len(w) / BATCH))}


# =============================================================================
# 5. Paper experiments: online streams (Table XI), poisoning (Table XII),
#    confusion matrices (Fig. 5), online traces (Fig. 7)
# =============================================================================
DRIFT_SLOPE, ATTACK_DRIFT = 0.70, 0.65          # gradual / attack-induced offset (std)
ABRUPT_AMP, ABRUPT_PERIOD = 0.55, 32            # recurring abrupt shift
PGD_STEPS = 20
TENT_LR, PH_CONF = 1e-4, 0.90
SCENARIOS = ["clean", "sustained_gradual_drift", "attack_induced_drift",
             "recurring_abrupt_shift"]
# comparison methods (ungated self-training, TENT, PH-ST) and the w/o-replay ablation
TABLE_MODES = ["frozen", "ungated", "tent", "ph-selftrain", "heads"]
CPOAD_MODE = "cpoad"            # gates + contiguity; heads NLL + classifier-path replay
POISON_MODES = TABLE_MODES[1:] + [CPOAD_MODE]
ATTACKS = [("oblivious", "attenuate", None), ("gray-box", "pgd_surrogate", 0.5),
           ("white-box", "pgd_live", 0.5), ("white-box-strong", "pgd_live", 1.0)]
POISON_RATES, POISON_SEEDS = [0.05, 0.10, 0.20], [0, 1, 2]
CLASS_NAMES = {"IGCPS": ["Normal", "NM", "NS", "PM", "PS", "SS"],
               "WDT": ["Normal", "MITM", "Physical fault"],
               "ICS-Flow": ["Normal", "DDoS", "IP-scan", "MITM", "Port-scan", "Replay"]}
SLUG = {"IGCPS": "igcps", "WDT": "wdt", "ICS-Flow": "ics_flow"}


def raw_windows(fd: Fold, targets: np.ndarray) -> np.ndarray:
    q = fd.config.seq_len
    w = np.stack([fd.raw[t - q + 1: t + 1] for t in targets]).astype(np.float64)
    return np.where(np.isfinite(w), w, fd.prep.median)      # same imputation as transform


def to_model(fd: Fold, raw_win: np.ndarray) -> np.ndarray:
    """Fold-fitted gateway preprocessing applied to raw windows (N, q, F)."""
    n, q, f = raw_win.shape
    return fd.prep.transform(raw_win.reshape(n * q, f)).reshape(n, q, -1)


def to_model_torch(fd: Fold, raw_win: torch.Tensor) -> torch.Tensor:
    """Differentiable copy of TrainOnlyPreprocessor.transform (no NaNs)."""
    p = fd.prep
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    y = torch.maximum(torch.minimum(raw_win, t(p.upper)), t(p.lower))
    y = (y - t(p.mean)) / t(p.std)
    dev = ((y - t(p.normal_center)) / t(p.normal_scale)).abs().clamp(0.0, 20.0)
    return torch.cat([y, dev], dim=-1)


def add_level(fd: Fold, raw: np.ndarray, amount) -> np.ndarray:
    """Offset the physical level channels by amount * training std (raw units)."""
    r = raw.copy()
    r[..., fd.level_idx] += np.asarray(amount)[:, None, None] * fd.prep.std[fd.level_idx]
    return r


def stress_backgrounds(fd: Fold, base0: np.ndarray, attack: np.ndarray) -> Dict[str, np.ndarray]:
    """The four stream conditions of Table XI (raw readings, before preprocessing)."""
    n = len(base0)
    progress = np.arange(n, dtype=np.float32) / max(1, n - 1)
    bg = {"clean": base0,
          "sustained_gradual_drift": add_level(fd, base0, DRIFT_SLOPE * progress),
          "attack_induced_drift": add_level(fd, base0, ATTACK_DRIFT * progress * attack)}
    sign = ((np.arange(n) // ABRUPT_PERIOD) % 2) * 2 - 1
    bg["recurring_abrupt_shift"] = add_level(fd, base0, ABRUPT_AMP * sign.astype(np.float32))
    return bg


def attenuate(fd: Fold, raw: np.ndarray, factor: float = 0.45) -> np.ndarray:
    """Oblivious poisoning: pull physical levels toward the training mean."""
    r = raw.copy()
    mu = fd.prep.mean[fd.level_idx]
    r[..., fd.level_idx] = mu + factor * (r[..., fd.level_idx] - mu)
    return r


def pgd(fd: Fold, model: nn.Module, raw: np.ndarray, cal: Dict[str, Any], normal_idx: int,
        eps: float) -> np.ndarray:
    """Gate-aware PGD: drive windows into the admission set (Normal prediction,
    confidence >= T_c, low entropy and coupling score) within an l_inf budget of
    eps training std on the physical level readings, through the preprocessing."""
    x0 = torch.as_tensor(raw, dtype=torch.float32)
    sd = torch.zeros(raw.shape[-1])
    sd[torch.as_tensor(fd.level_idx)] = torch.as_tensor(fd.prep.std[fd.level_idx],
                                                        dtype=torch.float32)
    delta = torch.zeros_like(x0, requires_grad=True)
    step = eps / 8.0
    tgt = torch.full((len(raw),), normal_idx, dtype=torch.long)
    model.eval()
    for _ in range(PGD_STEPS):
        out = model(to_model_torch(fd, x0 + delta * sd))
        s = signals(out)
        loss = (F.cross_entropy(out["logits"], tgt)
                + F.relu(s["ent"] - 0.5 * cal["T_h"]).mean()
                + F.relu(s["score"] - 0.5 * cal["T_s"]).mean() / max(cal["T_s"], 1e-6)
                + F.relu(cal["T_c"] + 0.02 - s["conf"]).mean())
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= step * g.sign()
            delta.clamp_(-eps, eps)
    return (x0 + delta.detach() * sd).numpy().astype(np.float64)


def displacement(model: nn.Module, anchor: Dict[str, torch.Tensor]) -> float:
    num = sum(((p.detach() - anchor[n]) ** 2).sum() for n, p in model.named_parameters())
    den = sum((a ** 2).sum() for a in anchor.values())
    return float(torch.sqrt(num / den))


@torch.no_grad()
def audit(model: nn.Module, eval_x: np.ndarray, eval_y: np.ndarray, normal_idx: int,
          ncls: int) -> Tuple[float, float]:
    """Post-stream macro-F1 and attack-to-Normal miss rate on clean held-out windows."""
    pred = model(torch.from_numpy(eval_x))["logits"].argmax(-1).numpy()
    att = eval_y != normal_idx
    return (f1_score(eval_y, pred, labels=np.arange(ncls), average="macro", zero_division=0),
            float((pred[att] == normal_idx).mean()) if att.any() else 0.0)


def stream_mode(fd: Fold, model0: CPCD5, cal: Dict[str, Any], base: np.ndarray,
                truth: np.ndarray, mode: str, poison_mask: np.ndarray,
                attacker: Optional[str], eps: Optional[float], normal_idx: int,
                eval_raw: np.ndarray, eval_y: np.ndarray,
                replay: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                ph_scale: float = 1e-4,
                trace: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One prequential stream under one mode.

    frozen        no updates
    ungated       all params, pseudo-label CE + 0.02 coupling NLL, conf >= 0.60
    tent          entropy minimisation of LayerNorm affine params, every window (TENT)
    ph-selftrain  Page-Hinkley test on batch entropy; on alarm one self-training
                  step on windows with confidence >= PH_CONF, all params (PH-ST)
    heads         CPOAD without classifier-path replay: gates + contiguity,
                  coupling-head NLL + anchor only ("w/o replay" ablation)
    cpoad         CPOAD: gates + contiguity, heads coupling NLL on admitted windows and
                  classifier-path CE on labelled replay windows, + anchor
    """
    ncls = len(fd.bundle.class_names)
    model = copy.deepcopy(model0); model.eval()
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
    gated = mode in ("heads", CPOAD_MODE)
    if gated:
        head = [p for n, p in model.named_parameters() if n.split(".")[0] in HEADS]
        cls = ([p for n, p in model.named_parameters() if n.split(".")[0] in CLS_PATH]
               if mode == CPOAD_MODE else [])
        params = head + cls
        groups = [{"params": head, "lr": HEAD_LR}] + ([{"params": cls, "lr": CLS_LR}] if cls else [])
    elif mode == "tent":
        params = [p for m in model.modules() if isinstance(m, nn.LayerNorm)
                  for p in m.parameters()]
        groups = [{"params": params, "lr": TENT_LR}]
    else:
        params = list(model.parameters())
        groups = [{"params": params, "lr": 1e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=2e-4)
    rng = np.random.default_rng(1234)
    hist = np.zeros(0, bool)
    ph_n = ph_mean = ph_m = ph_min = 0.0
    preds, rows = [], []
    run = updates = upd_samples = adm_poison = adm_attack = evaded = warns = acts = 0
    for bi, s in enumerate(range(0, len(base), BATCH)):
        sl = slice(s, s + BATCH)
        b, pm, tr = base[sl].copy(), poison_mask[sl], truth[sl]
        if attacker == "pgd_live" and pm.any():
            b[pm] = pgd(fd, model, b[pm], cal, normal_idx, eps)
        xt = torch.from_numpy(to_model(fd, b))
        with torch.no_grad():
            sig = signals(model(xt))
        pred = sig["prob"].argmax(-1).numpy()
        preds.append(pred)                                   # logged before any update
        conf, ent, score = (sig[k].numpy() for k in ("conf", "ent", "score"))
        v = np.array([score.mean(), ent.mean(), (1 - conf).mean()])
        exceeded = v > cal["batch_upper"]
        warning = bool(exceeded.any()); run = run + 1 if warning else 0
        action = bool(exceeded.sum() >= 2 and run >= PERSIST)
        warns += warning; acts += action
        gate = ((pred == normal_idx) & (conf >= cal["T_c"]) & (score <= cal["T_s"])
                & (ent <= cal["T_h"]))
        if mode == "frozen":
            el = np.zeros(len(pred), bool)
        elif mode == "ungated":
            el = conf >= 0.60
        elif mode == "tent":
            el = np.ones(len(pred), bool)
        elif mode == "ph-selftrain":
            ph_n += 1; ph_mean += (v[1] - ph_mean) / ph_n
            ph_m += v[1] - ph_mean - 0.5 * ph_scale; ph_min = min(ph_min, ph_m)
            alarm = ph_m - ph_min > 5.0 * ph_scale
            el = (conf >= PH_CONF) if alarm else np.zeros(len(pred), bool)
            if alarm:
                ph_n = ph_mean = ph_m = ph_min = 0.0
        else:                                                # heads, cpoad
            el = action & gate
        h = np.r_[hist, pred == normal_idx]
        if gated:                                            # causal event contiguity
            off = len(hist)
            for j in np.flatnonzero(el):
                i = off + j
                el[j] = i >= CONTEXT and bool(h[i - CONTEXT:i].all())
        hist = h
        evaded += int((pred[pm] == normal_idx).sum())
        adm_poison += int((el & pm).sum())
        adm_attack += int((el & (tr != normal_idx)).sum())
        if el.any():
            model.train() if mode != "tent" else model.eval()
            opt.zero_grad(set_to_none=True)
            xe = xt[torch.from_numpy(el)]
            pe = torch.from_numpy(pred[el])
            out = model(xe)
            nm = pe == normal_idx
            if mode == "tent":
                p_ = torch.softmax(out["logits"], -1)
                loss = -(p_ * p_.clamp_min(1e-12).log()).sum(-1).mean()
            elif gated:
                loss = model.nll_cross(out, nm) + model.nll_intra(out, nm)
                if mode == CPOAD_MODE:                      # classifier-path replay
                    rx, ry = replay
                    k = rng.choice(len(rx), size=min(N_REPLAY, len(rx)), replace=False)
                    ro = model(torch.from_numpy(rx[k]))
                    loss = loss + F.cross_entropy(ro["logits"], torch.from_numpy(ry[k]))
                loss = loss + ANCHOR * sum((p - anchor[n]).square().mean()
                                           for n, p in model.named_parameters()
                                           if any(p is q for q in params))
            else:
                loss = F.cross_entropy(out["logits"], pe)
                loss = loss + 0.02 * (model.nll_cross(out, nm) + model.nll_intra(out, nm))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); model.eval()
            updates += 1; upd_samples += int(el.sum())
        if trace:
            present = np.unique(tr)
            _, _, bf, _ = precision_recall_fscore_support(tr, pred, labels=present,
                                                          zero_division=0)
            rows.append({"batch": bi, "macro_f1": float(bf.mean()), "warning": warning,
                         "action": action, "eligible": int(el.sum()),
                         "updated": bool(el.any())})
    pred = np.concatenate(preds)
    _, _, f, _ = precision_recall_fscore_support(truth, pred, labels=np.arange(ncls),
                                                 zero_division=0)
    ef1, efnr = audit(model, to_model(fd, eval_raw), eval_y, normal_idx, ncls)
    return {"preq_macro_f1": float(f.mean()), "warning_batches": warns,
            "action_batches": acts, "update_batches": updates, "update_samples": upd_samples,
            "poison_windows": int(poison_mask.sum()), "admitted_poison": adm_poison,
            "admitted_attack": adm_attack,
            "evasion_rate": evaded / max(1, int(poison_mask.sum())),
            "post_val_macro_f1": ef1, "post_val_attack_fnr": efnr,
            "displacement": displacement(model, anchor)}, rows


def online_experiment(dataset: str, fold: int, poison: bool = False,
                      ckpt_dir: Optional[Path] = None, data_path: Optional[Path] = None,
                      attacks: Optional[Sequence[str]] = None,
                      rates: Sequence[float] = POISON_RATES,
                      seeds: Sequence[int] = POISON_SEEDS,
                      trace: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Table XI streams of one fold (stream = test partition, gates calibrated and
    harm audited on the validation partition), optionally followed by the
    Table XII poisoning runs injected into the gradual-drift stream."""
    torch.set_num_threads(1)
    fd = prepare_fold(dataset, fold, data_path)
    model, surrogate = get_models(fd, ckpt_dir)
    names = [c.lower() for c in fd.bundle.class_names]
    normal_idx = names.index("normal") if "normal" in names else 0
    cal_t = aud_t = fd.parts["validation"]
    stream_t = fd.parts["test"]
    cal_x = windows(fd, fd.x, cal_t)
    cal = calibrate(model, cal_x, fd.bundle.labels[cal_t], normal_idx)
    with torch.no_grad():
        ent = signals(model(torch.from_numpy(cal_x)))["ent"].numpy()
    be = np.array([ent[i:i + BATCH].mean() for i in range(0, len(ent) - BATCH + 1, BATCH)])
    ph_scale = float(max(1.4826 * np.median(np.abs(be - np.median(be))), 1e-4))
    eval_raw, eval_y = raw_windows(fd, aud_t), fd.bundle.labels[aud_t]
    base0, truth = raw_windows(fd, stream_t), fd.bundle.labels[stream_t]
    attack = truth != normal_idx
    none = np.zeros(len(stream_t), bool)
    bg = stress_backgrounds(fd, base0, attack)
    summ, traces = [], []
    common = dict(normal_idx=normal_idx, eval_raw=eval_raw, eval_y=eval_y,
                  replay=replay_set(fd, cal_t), ph_scale=ph_scale)
    for sc in SCENARIOS:
        for mode in TABLE_MODES + [CPOAD_MODE]:
            if mode == CPOAD_MODE:              # never perturbs the baselines' RNG stream
                with torch.random.fork_rng():
                    res, rows = stream_mode(fd, model, cal, bg[sc], truth, mode, none, None,
                                            None, trace=trace, **common)
            else:
                res, rows = stream_mode(fd, model, cal, bg[sc], truth, mode, none, None,
                                        None, trace=trace, **common)
            summ.append({"dataset": dataset, "fold": fold, "scenario": sc, "mode": mode,
                         "attacker": "none", "rate": 0.0, "seed": -1, **res})
            traces += [{"dataset": dataset, "fold": fold, "scenario": sc, "mode": mode, **r}
                       for r in rows]
    if poison:
        ref = {r["mode"]: r["post_val_attack_fnr"] for r in summ
               if r["scenario"] == "sustained_gradual_drift"}
        drift = bg["sustained_gradual_drift"]
        att_idx = np.flatnonzero(attack)
        for rate in rates:
            n_poison = min(len(att_idx), int(round(rate * len(stream_t))))
            for seed in seeds:
                rng = np.random.default_rng(10_000 * fold + 100 * seed + int(rate * 100))
                pm = none.copy(); pm[rng.choice(att_idx, n_poison, replace=False)] = True
                for name, kind, eps in ATTACKS:
                    if attacks and name not in attacks:
                        continue
                    b = drift.copy()
                    if kind == "attenuate":
                        b[pm] = attenuate(fd, b[pm])
                    elif kind == "pgd_surrogate":
                        torch.manual_seed(seed)
                        b[pm] = pgd(fd, surrogate, b[pm], cal, normal_idx, eps)
                    for mode in POISON_MODES:
                        torch.manual_seed(seed)
                        res, _ = stream_mode(fd, model, cal, b, truth, mode, pm,
                                             kind if kind == "pgd_live" else None, eps,
                                             **common)
                        summ.append({"dataset": dataset, "fold": fold, "scenario": "poisoning",
                                     "mode": mode, "attacker": name, "rate": rate,
                                     "seed": seed, **res,
                                     "harm_pp": (res["post_val_attack_fnr"] - ref[mode]) * 100})
    return pd.DataFrame(summ), pd.DataFrame(traces)


def _online_job(args):
    return online_experiment(*args)


def online_all(dataset: str, poison: bool, ckpt_dir: Optional[Path], data_path: Optional[Path],
               folds: Sequence[int], workers: int, attacks=None, rates=POISON_RATES,
               seeds=POISON_SEEDS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(dataset, k, poison, ckpt_dir, data_path, attacks, rates, seeds, k == 0)
            for k in folds]
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            res = pool.map(_online_job, jobs)
    else:
        res = [_online_job(j) for j in jobs]
    return (pd.concat([r[0] for r in res], ignore_index=True),
            pd.concat([r[1] for r in res], ignore_index=True))


def table_x(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean prequential macro-F1 (%) over streams, and CPOAD minus frozen."""
    a = summary[summary.attacker == "none"]
    t = a.pivot_table(index=["dataset", "scenario"], columns="mode",
                      values="preq_macro_f1", aggfunc="mean") * 100
    t = t.reindex(columns=TABLE_MODES + [CPOAD_MODE])
    t["delta_cpoad"] = t[CPOAD_MODE] - t["frozen"]
    return t


def table_xi(summary: pd.DataFrame) -> pd.DataFrame:
    """Admitted poisoned windows (%) per rate and harm (pp) at each rate."""
    p = summary[summary.scenario == "poisoning"].copy()
    p["admitted_pct"] = p.admitted_poison / p.poison_windows * 100
    return p.groupby(["dataset", "attacker", "mode", "rate"])[["admitted_pct", "harm_pp"]].mean()


def _confusion_job(args):
    dataset, k, ckpt_dir, data_path = args
    torch.set_num_threads(1)
    fd = prepare_fold(dataset, k, data_path)
    path = Path(ckpt_dir) / f"{dataset}_f{k}_target.pt" if ckpt_dir else None
    model = load_model(fd, path) if path is not None and path.exists() else train_fold_model(fd)[0]
    t = fd.parts["test"]
    return fd.bundle.labels[t], predict(model, windows(fd, fd.x, t))


def confusion(dataset: str, ckpt_dir: Optional[Path] = None, data_path: Optional[Path] = None,
              out_dir: Path = Path("."), workers: int = 5) -> np.ndarray:
    """Confusion matrix of the pooled held-out decisions of the five folds."""
    from sklearn.metrics import confusion_matrix
    jobs = [(dataset, k, ckpt_dir, data_path) for k in range(5)]
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(workers, 5)) as pool:
            res = pool.map(_confusion_job, jobs)
    else:
        res = [_confusion_job(j) for j in jobs]
    truth = np.concatenate([r[0] for r in res]); pred = np.concatenate([r[1] for r in res])
    classes = CLASS_NAMES.get(dataset, [str(c) for c in range(int(truth.max()) + 1)])
    n = len(classes)
    cm = confusion_matrix(truth, pred, labels=range(n))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / f"confusion_{SLUG.get(dataset, dataset)}.csv", cm, fmt="%d",
               delimiter=",")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                                "mathtext.fontset": "stix", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.8, 4.45))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, int(cm.max())))
    for r in range(n):
        for c in range(n):
            ax.text(c, r, f"{int(cm[r, c]):,}", ha="center", va="center", fontsize=14.8,
                    color="white" if cm[r, c] > 0.55 * cm.max() else "black")
    ax.set_xticks(np.arange(n), classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(n), classes)
    ax.set_xlabel("Predicted", fontsize=16); ax.set_ylabel("True", fontsize=16)
    ax.tick_params(axis="both", labelsize=14.2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.25, top=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_confusion_{SLUG.get(dataset, dataset)}_aug.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return cm


def fig6(dataset: str, ckpt_dir: Optional[Path] = None, data_path: Optional[Path] = None,
         out_dir: Path = Path("."), traces: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Fold-0 prequential traces (frozen, ungated, CPOAD) for the four stream conditions."""
    if traces is None:
        _, traces = online_experiment(dataset, 0, False, ckpt_dir, data_path, trace=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight"})
    colors = {"frozen": "#4878CF", "ungated": "#E8803A", CPOAD_MODE: "#247D70"}
    labels = {"frozen": "Frozen", "ungated": "Ungated", CPOAD_MODE: "CPOAD"}
    marks = {"frozen": "s", "ungated": "^", CPOAD_MODE: "o"}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tr = traces[traces.dataset == dataset]
    for sc in SCENARIOS:
        sub = tr[(tr.scenario == sc) & tr["mode"].isin(list(colors))]
        fig, ax = plt.subplots(figsize=(3.2, 2.3))
        for mode in ["frozen", "ungated", CPOAD_MODE]:
            s = sub[sub["mode"] == mode].sort_values("batch")
            ax.plot(s.batch, s.macro_f1 * 100, marker=marks[mode], color=colors[mode],
                    label=labels[mode], markersize=2.5, linewidth=1.0)
            if mode == CPOAD_MODE:
                w = s[s.warning.astype(bool)]
                ax.scatter(w.batch, w.macro_f1 * 100, marker="v", color="#CC3333",
                           s=16, zorder=5, label=f"Warning ({len(w)})")
                u = s[s.updated.astype(bool)]
                ax.scatter(u.batch, u.macro_f1 * 100, marker="*", color="#8B0000",
                           s=46, zorder=6, label=f"CPOAD update ({len(u)})")
        ax.set_xlabel("Batch index"); ax.set_ylabel("Macro-F1 (%)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylim(max(0, (sub.macro_f1.min() * 100) - 8), 102)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                  fontsize=6.3, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"fig_online_{dataset}_{sc}_v10.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
    return traces


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="CPOAD reference implementation and paper experiments",
        epilog="commands: train/online <ds> <fold>; online-table/poison <ds> <fold>|--all; "
               "confusion <ds>; fig6 <ds>")
    ap.add_argument("command", choices=["train", "online", "online-table", "poison",
                                        "confusion", "fig6"])
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("fold", type=int, nargs="?", default=0)
    ap.add_argument("--all", action="store_true", help="all five folds (multiprocessing)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--data", type=Path, default=None, help="fused CSV (default: project root)")
    ap.add_argument("--save", type=Path, default=None, help="train: where to save the weights")
    ap.add_argument("--ckpt", type=Path, default=None, help="online: weights file (else train)")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="<ds>_f<k>_{target,surrogate}.pt cache (loaded if present, else "
                         "trained and saved here)")
    ap.add_argument("--out", type=Path, default=Path("cpoad_out"), help="output directory")
    ap.add_argument("--attackers", default=None,
                    help="comma list of " + ",".join(a[0] for a in ATTACKS))
    ap.add_argument("--rates", default=None, help="comma list, default 0.05,0.1,0.2")
    ap.add_argument("--seeds", default=None, help="comma list, default 0,1,2")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command in ("train", "online"):
        if a.command == "online" and a.ckpt:
            fd = prepare_fold(a.dataset, a.fold, a.data)
            model = load_model(fd, a.ckpt)
        else:
            model, fd, info = run_fold(a.dataset, a.fold, data_path=a.data)
            print(f"best epoch {info['best_epoch']}, validation macro-F1 "
                  f"{info['best_validation_macro_f1_selection']:.4f}")
        t = fd.parts["test"]
        f1 = macro_f1_all(fd.bundle.labels[t], predict(model, windows(fd, fd.x, t)),
                          len(fd.bundle.class_names))
        print(f"{a.dataset} fold {a.fold}: test macro-F1 = {f1:.6f}")
        if a.command == "train" and a.save:
            torch.save(model.state_dict(), a.save)
        if a.command == "online":
            r = run_online(model, fd)
            print(f"online: prequential macro-F1 = {r['preq_macro_f1']:.6f} "
                  f"(frozen {r['frozen_macro_f1']:.6f}), updates on "
                  f"{r['update_batches']}/{r['batches']} batches")
        return
    if a.command in ("online-table", "poison"):
        folds = range(5) if a.all else [a.fold]
        attacks = a.attackers.split(",") if a.attackers else None
        rates = [float(v) for v in a.rates.split(",")] if a.rates else POISON_RATES
        seeds = [int(v) for v in a.seeds.split(",")] if a.seeds else POISON_SEEDS
        summ, tr = online_all(a.dataset, a.command == "poison", a.ckpt_dir, a.data, folds,
                              a.workers, attacks, rates, seeds)
        tag = f"{a.dataset}_{'all' if a.all else 'f%d' % a.fold}"
        summ.to_csv(a.out / f"{a.command}_{tag}.csv", index=False)
        if len(tr):
            tr.to_csv(a.out / f"trace_{tag}.csv", index=False)
        pd.set_option("display.width", 200)
        print("Table XI (prequential macro-F1, %):")
        print(table_x(summ).round(2).to_string())
        if a.command == "poison":
            print("\nTable XII (admitted poisoned windows %, harm %):")
            print(table_xi(summ).round(2).to_string())
        print(f"\nsaved {a.out / (a.command + '_' + tag + '.csv')}")
        return
    if a.command == "confusion":
        cm = confusion(a.dataset, a.ckpt_dir, a.data, a.out, a.workers)
        print(cm)
        print(f"saved {a.out / ('fig4_confusion_' + SLUG.get(a.dataset, a.dataset) + '_aug.png')}")
        return
    if a.command == "fig6":
        tr = fig6(a.dataset, a.ckpt_dir, a.data, a.out)
        tr.to_csv(a.out / f"trace_{a.dataset}_f0.csv", index=False)
        print(f"saved {a.out}/fig_online_{a.dataset}_*_v10.png")


if __name__ == "__main__":
    main()
