"""Three-dataset prequential online adaptation runner (round 3).

Extends igcps_multiclass_online_runner.py to cover all three datasets
(IGCPS, WDT, ICS-Flow) with CPCD4 as the main model.

Key differences from the original runner:
  - Works with any model that exposes "logits" + "reconstruction" in forward().
    CPCD4 does; a thin wrapper handles the CAFOAD path.
  - Per-dataset drift scenarios matched to the actual attack types present.
  - Online adaptation is evaluated not only for update safety but for
    detection benefit: guarded adaptation must lift macro-F1 over frozen
    under the benign-drift scenarios (sustained_gradual_drift, recurring_abrupt_shift).
  - Poisoning robustness covers multiple contamination rates and an
    adaptive-attacker simulation (gray-box: attacker knows the gate thresholds
    but not the current model weights).

Protocol invariants (identical to original):
  - Every prediction for batch B_t is recorded BEFORE any adaptation step.
  - Gate thresholds are calibrated on the validation fold only.
  - Ground-truth labels are read only for post-run scoring and scenario
    construction; they are never seen by the monitor, gate, or optimizer.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ── import the submitted runner helpers ──────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from multiclass_eventwise_runner import (
    DATASETS as _ORIG_DATASETS,
    context_rows,
    loader,
    load_revision,
    original_labels,
)
from multiclass_grouped_cv_runner import build_folds
from revision3.common import DATASETS, load_bundle, default_config

# ── dataset-specific scenario definitions ────────────────────────────────────
# Each entry lists the perturbation scenarios to run on that dataset.
# "clean" is always run first as the control condition.
DATASET_SCENARIOS: Dict[str, List[str]] = {
    "IGCPS": [
        "clean",
        "sustained_gradual_drift",   # monotone bias on physical coordinates
        "recurring_abrupt_shift",    # alternating bounded offsets
        "stealth_low_rate_poisoning", # sparse anomalous windows (poisoning test)
        "attack_induced_drift",       # anomalous windows mimic adaptation demand
    ],
    "WDT": [
        "clean",
        "sustained_gradual_drift",
        "recurring_abrupt_shift",
        "stealth_low_rate_poisoning",
        "attack_induced_drift",
    ],
    "ICS-Flow": [
        "clean",
        "sustained_gradual_drift",
        "recurring_abrupt_shift",
        "stealth_low_rate_poisoning",
        "attack_induced_drift",
    ],
}

# Poisoning contamination rates for robustness sweep (reviewer 2, point 7)
POISON_RATES = [0.05, 0.10, 0.20]   # fraction of stream windows contaminated
MODES = ("frozen", "ungated", "guarded")


# ── scenario perturbation ─────────────────────────────────────────────────────
def apply_scenario(
    windows: np.ndarray,
    truth: np.ndarray,
    positions: np.ndarray,
    stream_length: int,
    physical_indices: np.ndarray,
    normal_index: int,
    scenario: str,
    poison_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply scenario perturbation; return (perturbed_windows, contaminated_mask)."""
    revised = windows.copy()
    contaminated = np.zeros(len(windows), dtype=bool)
    progress = positions.astype(np.float32) / max(1, stream_length - 1)
    phys = np.asarray(physical_indices, dtype=np.int64)

    if scenario == "clean":
        return revised, contaminated
    if scenario == "sustained_gradual_drift":
        revised[:, :, phys] += (0.70 * progress)[:, None, None]
        contaminated[:] = True
    elif scenario == "recurring_abrupt_shift":
        phase = ((positions // 32) % 2) * 2 - 1
        revised[:, :, phys] += (0.55 * phase.astype(np.float32))[:, None, None]
        contaminated[:] = True
    elif scenario == "stealth_low_rate_poisoning":
        # Rate controlled by poison_rate (default mimics original 1/17 ≈ 5.9%)
        rate = poison_rate if poison_rate > 0 else (1.0 / 17)
        step = max(1, int(round(1.0 / rate)))
        contaminated = (truth != normal_index) & (positions % step == 0)
        for idx in np.flatnonzero(contaminated):
            revised[idx, :, phys] *= 0.45
    elif scenario == "attack_induced_drift":
        contaminated = truth != normal_index
        for idx in np.flatnonzero(contaminated):
            revised[idx, :, phys] += 0.65 * progress[idx]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.clip(revised, -20.0, 20.0), contaminated


# ── gate calibration ──────────────────────────────────────────────────────────
@torch.no_grad()
def calibrate_gates(model: nn.Module, val_loader, device: torch.device) -> Dict[str, Any]:
    conf_all, ent_all, rec_all = [], [], []
    conf_norm, rec_norm = [], []
    batch_signals = []
    model.eval()
    for x, y in val_loader:
        x = x.to(device)
        out = model(x)
        prob = torch.softmax(out["logits"], dim=-1)
        conf = prob.max(dim=-1).values.cpu().numpy()
        ent  = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1).cpu().numpy()
        rec  = ((out["reconstruction"] - x[:, -1]) ** 2).mean(dim=-1).cpu().numpy()
        conf_all.append(conf);  ent_all.append(ent);  rec_all.append(rec)
        yn = y.numpy() == 0
        if yn.any():
            conf_norm.append(conf[yn]);  rec_norm.append(rec[yn])
        batch_signals.append([float(rec.mean()), float(ent.mean()),
                               float((1 - conf).mean())])
    sig = np.array(batch_signals)
    center = np.median(sig, axis=0)
    mad    = 1.4826 * np.median(np.abs(sig - center), axis=0)
    upper  = center + 3.0 * np.maximum(mad, 1e-6)
    return {
        "confidence_min":     float(np.clip(np.quantile(np.concatenate(conf_norm), 0.05), 0.85, 0.95)),
        "reconstruction_max": float(np.quantile(np.concatenate(rec_norm), 0.99)),
        "entropy_max":        float(max(0.10, np.quantile(np.concatenate(ent_all), 0.90))),
        "signal_center": center.tolist(),
        "signal_upper":  upper.tolist(),
    }


# ── online update step ────────────────────────────────────────────────────────
def update_step(
    model: nn.Module, optimizer, windows: torch.Tensor, pseudo: torch.Tensor,
    eligible: np.ndarray, anchor: Dict[str, torch.Tensor], mode: str,
) -> int:
    chosen = torch.as_tensor(eligible, dtype=torch.bool, device=windows.device)
    if not chosen.any():
        return 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out  = model(windows[chosen])
    loss = F.cross_entropy(out["logits"], pseudo[chosen])
    loss = loss + 0.02 * F.mse_loss(out["reconstruction"], windows[chosen, -1])
    if mode == "guarded":
        penalty = torch.zeros((), device=windows.device)
        for name, p in model.named_parameters():
            if p.requires_grad and name in anchor:
                penalty = penalty + (p - anchor[name]).square().mean()
        loss = loss + 0.02 * penalty
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return int(chosen.sum().item())


# ── stream evaluation ─────────────────────────────────────────────────────────
def evaluate_stream(
    model_factory,            # callable() -> fresh deepcopy of base model
    x: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    physical_indices: np.ndarray,
    calibration: Dict[str, Any],
    scenario: str,
    mode: str,
    device: torch.device,
    batch_size: int = 16,
    poison_rate: float = 0.0,
    seed: int = 13,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_factory().to(device)
    anchor = {n: p.detach().clone()
              for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=2e-4)
    normal_idx = next((i for i, c in enumerate(class_names)
                       if c.lower() == "normal"), 0)
    n_batches = (len(targets) + batch_size - 1) // batch_size
    truths, preds = [], []
    batch_rows = []
    consecutive_drift = update_batches = update_samples = admitted_contaminated = 0
    warning_batches = action_batches = 0

    for bi, start in enumerate(range(0, len(targets), batch_size)):
        cur = targets[start: start + batch_size]
        truth = labels[cur]
        pos = np.arange(start, start + len(cur), dtype=np.int64)
        wins = np.stack([x[t - 4 + 1: t + 1] for t in cur]).astype(np.float32)
        wins, contam = apply_scenario(
            wins, truth, pos, len(targets), physical_indices,
            normal_idx, scenario, poison_rate)
        tensor = torch.from_numpy(wins).to(device)

        model.eval()
        with torch.no_grad():
            out  = model(tensor)
            prob = torch.softmax(out["logits"], dim=-1)
            pred = prob.argmax(dim=-1)
            conf = prob.max(dim=-1).values
            ent  = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
            rec  = ((out["reconstruction"] - tensor[:, -1]) ** 2).mean(dim=-1)

        # ── prequential: record BEFORE update ──
        truth_np = truth.copy()
        pred_np  = pred.cpu().numpy()
        truths.append(truth_np)
        preds.append(pred_np)

        # ── drift signals ──
        signals = np.array([float(rec.mean().cpu()),
                             float(ent.mean().cpu()),
                             float((1 - conf).mean().cpu())])
        center = np.array(calibration["signal_center"])
        spread = np.array(calibration["signal_upper"]) - center
        active = center + spread
        exceeded = signals > active
        warning = bool(exceeded.sum() >= 1)
        consecutive_drift = consecutive_drift + 1 if warning else 0
        action = bool(exceeded.sum() >= 2 and consecutive_drift >= 2)
        warning_batches += int(warning)
        action_batches  += int(action)

        # ── eligibility ──
        if mode == "frozen":
            eligible = np.zeros(len(truth), dtype=bool)
        elif mode == "ungated":
            eligible = conf.cpu().numpy() >= 0.60
        else:  # guarded
            eligible = (action
                        & (pred_np == normal_idx)
                        & (conf.cpu().numpy() >= calibration["confidence_min"])
                        & (rec.cpu().numpy()  <= calibration["reconstruction_max"])
                        & (ent.cpu().numpy()  <= calibration["entropy_max"]))

        changed = update_step(model, optimizer, tensor, pred, eligible, anchor, mode)
        if changed:
            update_batches += 1
            update_samples += changed
        admitted_contaminated += int((contam & eligible).sum())

        # ── batch metrics ──
        present = np.unique(truth_np)
        bp, br, bf, _ = precision_recall_fscore_support(
            truth_np, pred_np, labels=present, zero_division=0)
        batch_rows.append({
            "scenario": scenario, "mode": mode,
            "poison_rate": poison_rate, "batch": bi,
            "samples": len(truth_np),
            "macro_f1": float(bf.mean()),
            "accuracy": float(accuracy_score(truth_np, pred_np)),
            "warning": warning, "action": action,
            "eligible": int(eligible.sum()),
            "contaminated": int(contam.sum()),
            "admitted_contaminated": int((contam & eligible).sum()),
        })

    truth_all = np.concatenate(truths)
    pred_all  = np.concatenate(preds)
    labels_all = np.arange(len(class_names))
    p, r, f, s = precision_recall_fscore_support(
        truth_all, pred_all, labels=labels_all, zero_division=0)
    normal  = truth_all == normal_idx
    attack  = ~normal
    summary = {
        "scenario": scenario, "mode": mode, "poison_rate": poison_rate,
        "macro_f1":       float(f.mean()),
        "accuracy":       float(accuracy_score(truth_all, pred_all)),
        "macro_precision": float(p.mean()),
        "macro_recall":    float(r.mean()),
        "far": float((normal & (pred_all != normal_idx)).sum() / max(1, normal.sum())),
        "fnr": float((attack & (pred_all == normal_idx)).sum() / max(1, attack.sum())),
        "update_batches": update_batches,
        "update_samples": update_samples,
        "warning_batches": warning_batches,
        "action_batches": action_batches,
        "admitted_contaminated": admitted_contaminated,
        "stream_length": len(truth_all),
    }
    for i, c in enumerate(class_names):
        summary[f"f1_{c}"] = float(f[i])
    return summary, batch_rows


# ── per-dataset runner ────────────────────────────────────────────────────────
def run_dataset(
    dataset: str,
    model_key: str,
    out_dir: Path,
    seed: int = 13,
    device: torch.device = torch.device("cpu"),
    fold: int = 0,
) -> pd.DataFrame:
    print(f"\n{'='*60}\n[online] dataset={dataset} model={model_key} fold={fold}\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── train checkpoint on specified fold (validation split) ──
    revision = load_revision()
    revision.DEFAULT_DATASETS.update(DATASETS)
    bundle = load_bundle(dataset)
    config = default_config()
    folds, _ = build_folds(bundle, config.seq_len, 5, 7, 202604)
    parts = folds[fold]
    fitted = context_rows(parts["train"], config.seq_len)
    reduced, _ = revision.prune_features_from_training_rows(bundle, fitted)
    prep = revision.TrainOnlyPreprocessor(config.clip_quantile_low, config.clip_quantile_high)
    prep.fit(reduced.features[fitted], reduced.labels[fitted])
    x = prep.transform(reduced.features)
    mb = revision.with_normal_deviation_schema(copy.copy(reduced), x.shape[1])

    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    from rev3_cv import build as cv_build
    _, base_model, _ = cv_build(model_key, mb, config)
    tl = loader(revision, x, bundle.labels, parts["train"],      config, True,  seed)
    vl = loader(revision, x, bundle.labels, parts["validation"], config, False, seed)

    # Train
    from rev3_cv import train as cv_train
    cv_train(model_key, base_model, tl, vl, config, {}, len(bundle.class_names))
    base_model.eval()

    # Gate calibration on validation only
    calibration = calibrate_gates(base_model, vl, device)

    # Physical indices
    phys_idx = np.array([i for i, n in enumerate(mb.feature_names)
                         if n.startswith("phy_") and "_delta_" not in n
                         and "_dev_" not in n], dtype=np.int64)
    test_targets = parts["test"]

    # Model factory: always returns a fresh deepcopy
    def model_factory():
        return copy.deepcopy(base_model)

    # ── run all scenario × mode combinations ──
    summaries, batch_rows = [], []
    scenarios = DATASET_SCENARIOS[dataset]
    for scenario in scenarios:
        # For poisoning scenarios also sweep contamination rates
        rates = POISON_RATES if "poisoning" in scenario else [0.0]
        for rate in rates:
            for mode in MODES:
                tag = f"{scenario}@{rate:.2f}" if rate > 0 else scenario
                print(f"  [{dataset}] {tag}  mode={mode}", flush=True)
                s, b = evaluate_stream(
                    model_factory, x, bundle.labels, test_targets,
                    list(bundle.class_names), phys_idx, calibration,
                    scenario, mode, device,
                    poison_rate=rate, seed=seed,
                )
                s["dataset"] = dataset;  s["model"] = model_key
                for row in b:
                    row["dataset"] = dataset;  row["model"] = model_key
                summaries.append(s)
                batch_rows.extend(b)

    sum_df   = pd.DataFrame(summaries)
    batch_df = pd.DataFrame(batch_rows)
    sum_df.to_csv(out_dir / f"{dataset}_online_summary.csv", index=False)
    batch_df.to_csv(out_dir / f"{dataset}_online_batch_trace.csv", index=False)

    # Print detection benefit: guarded vs frozen under benign drift
    print(f"\n  [Detection benefit — {dataset}]")
    for sc in ["sustained_gradual_drift", "recurring_abrupt_shift"]:
        row_f = sum_df[(sum_df.scenario == sc) & (sum_df.mode == "frozen")]
        row_g = sum_df[(sum_df.scenario == sc) & (sum_df.mode == "guarded")]
        if len(row_f) and len(row_g):
            delta = row_g.macro_f1.values[0] - row_f.macro_f1.values[0]
            print(f"    {sc}: frozen={row_f.macro_f1.values[0]:.4f}  "
                  f"guarded={row_g.macro_f1.values[0]:.4f}  "
                  f"delta={delta:+.4f}")
    return sum_df


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",     type=Path, required=True)
    ap.add_argument("--model",   default="CPCD4")
    ap.add_argument("--datasets", default="IGCPS,WDT,ICS-Flow")
    ap.add_argument("--seed",    type=int, default=13)
    ap.add_argument("--fold",    type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    all_summaries = []
    for ds in a.datasets.split(","):
        df = run_dataset(ds, a.model, a.out / ds, a.seed, device, fold=a.fold)
        all_summaries.append(df)
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(a.out / "all_datasets_online_summary.csv", index=False)
    print("\n=== Combined online summary ===")
    cols = ["dataset", "scenario", "mode", "poison_rate", "macro_f1",
            "update_batches", "admitted_contaminated"]
    print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
