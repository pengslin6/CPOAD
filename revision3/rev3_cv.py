"""Repeated purged grouped CV with retrained ablations (round 3).

Each job = (dataset, model, split_seed, fold).  The split rule is the
submitted ``build_folds`` with a different ``split_seed`` per repetition, so
every repetition is a fresh class-stratified assignment of event blocks.
Both validation and test macro-F1 are recorded; model development decisions
may read only the validation column.

Usage:
  python rev3_cv.py --out results/ablation --models CAFOAD-full,w/o-CAF \
      --datasets IGCPS --splits 202604,1,2,3,4 --workers 22
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

_BUNDLES: Dict[str, Any] = {}
_FOLDS: Dict[Tuple[str, int], Any] = {}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train(model_name: str, model: nn.Module, train_loader, validation_loader,
          config, flags: Dict[str, bool], n_classes: int) -> Dict[str, Any]:
    """Copy of the submitted ``train_model`` with ablation switches."""
    from common import revision
    device = torch.device("cpu")
    is_cafoad = isinstance(model, revision.CAFOAD)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(n_classes))
    if is_cafoad and not flags.get("no_residual"):
        rows, labels = [], []
        unique = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False)
        for bx, by in unique:
            rows.append(model.residual_context(bx))
            labels.append(by)
        wx, wy = torch.cat(rows), torch.cat(labels)
        counts = torch.bincount(wy, minlength=model.class_prototypes.shape[0]).float()
        wcrit = nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts.clamp_min(1.0)))
        opt = torch.optim.LBFGS(model.input_classifier.parameters(), lr=0.5, max_iter=80,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = wcrit(model.input_classifier(wx), wy)
            loss = loss + 1.0e-4 * torch.sum(model.input_classifier.weight ** 2)
            loss.backward()
            return loss

        opt.step(closure)
        for p in model.input_classifier.parameters():
            p.requires_grad_(False)
    loss_name = "CAFOAD" if is_cafoad and not flags.get("no_supcon") else model_name
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "loss"):
                loss = model.loss(x, y)  # CPCD: focal + Normal-only coupling NLL
            else:
                loss = revision.batch_loss(loss_name, model, x, y, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        yv, pv, _, _ = revision.predict_loader(model, validation_loader, device)
        score = revision.macro_f1_present(yv, pv)
        if score > best_f1 + 1e-4:
            best_f1, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    calibration = None
    if is_cafoad and not flags.get("no_prototype"):
        revision.fit_cafoad_prototypes(model, train_loader, device)
        calibration = revision.calibrate_cafoad_ensemble(model, validation_loader, device)
    return {"best_validation_macro_f1_selection": best_f1, "best_epoch": best_epoch,
            "epochs_executed": epoch + 1, "calibration": calibration}


def build(model_key: str, model_bundle, config):
    from common import revision
    from rev3_models import VARIANTS, CAFOADVariant
    from rev3_cpcd import CPCD_VARIANTS, build_cpcd
    from rev3_cpcd3 import CPCD3_VARIANTS, build_cpcd3
    from rev3_cpcd4 import CPCD4_VARIANTS, build_cpcd4
    from rev3_cpcd5 import CPCD5_VARIANTS, build_cpcd5
    # Shared hyperparameter alias table for all CPCD variants
    _alias = {"d": ("d_model", int), "l": ("lambda_nll", float),
              "i": ("lambda_intra", float),
              "det": ("detach_residual", lambda v: bool(int(v))),
              "do": ("dropout", float)}
    _alias5 = {**_alias, "c": ("lambda_contrastive", float)}
    _alias4 = {"d": ("d_model", int), "p": ("lambda_phys", float),
               "c": ("lambda_cyber", float), "do": ("dropout", float)}
    if model_key.split("@")[0] in CPCD5_VARIANTS:
        hp = {}
        if "@" in model_key:
            for item in model_key.split("@")[1].split(";"):
                k, v = item.split("=")
                hp[_alias5[k][0]] = _alias5[k][1](v)
        return model_key, build_cpcd5(model_key, model_bundle, config, **hp), {}
    if model_key.split("@")[0] in CPCD4_VARIANTS:
        hp = {}
        if "@" in model_key:
            for item in model_key.split("@")[1].split(";"):
                k, v = item.split("=")
                hp[_alias4[k][0]] = _alias4[k][1](v)
        return model_key, build_cpcd4(model_key, model_bundle, config, **hp), {}
    if model_key.split("@")[0] in CPCD3_VARIANTS:
        hp = {}
        if "@" in model_key:
            for item in model_key.split("@")[1].split(";"):
                k, v = item.split("=")
                hp[_alias[k][0]] = _alias[k][1](v)
        return model_key, build_cpcd3(model_key, model_bundle, config, **hp), {}
    if model_key.split("@")[0] in CPCD_VARIANTS:
        hp = {}
        if "@" in model_key:
            for item in model_key.split("@")[1].split(";"):
                k, v = item.split("=")
                hp[_alias[k][0]] = _alias[k][1](v)
        return model_key, build_cpcd(model_key, model_bundle, config, **hp), {}
    if model_key in VARIANTS:
        kwargs = revision.method_model_kwargs("CAFOAD", model_bundle, config)
        return "CAFOAD", CAFOADVariant(**kwargs, flags=VARIANTS[model_key]), VARIANTS[model_key]
    # New baselines: OmniAnomaly, DAGMM, OL-TSAD
    try:
        from rev3_new_baselines import NEW_MODELS, build_new_model
        if model_key in NEW_MODELS:
            return model_key, build_new_model(model_key, model_bundle, config), {}
    except ImportError:
        pass
    # New baselines batch 2: DeepSVDD, CNN-1D
    try:
        from rev3_new_baselines2 import NEW_MODELS_2, build_new_model_2
        if model_key in NEW_MODELS_2:
            return model_key, build_new_model_2(model_key, model_bundle, config), {}
    except ImportError:
        pass
    # New baselines batch 3: RAIID, AE-Stream
    try:
        from rev3_new_baselines3 import NEW_MODELS_3, build_new_model_3
        if model_key in NEW_MODELS_3:
            return model_key, build_new_model_3(model_key, model_bundle, config), {}
    except ImportError:
        pass
    # New baselines batch 4: Shallow-LSTM-AE, RandomProj-CLF
    try:
        from rev3_new_baselines4 import NEW_MODELS_4, build_new_model_4
        if model_key in NEW_MODELS_4:
            return model_key, build_new_model_4(model_key, model_bundle, config), {}
    except ImportError:
        pass
    extra = {}
    try:
        import rev3_new_models as nm  # optional, added later
        if model_key in nm.MODELS:
            factory, kwargs = nm.MODELS[model_key](model_bundle, config)
            return model_key, factory(**kwargs), nm.FLAGS.get(model_key, {})
    except ImportError:
        pass
    kwargs = revision.method_model_kwargs(model_key, model_bundle, config)
    return model_key, revision.MODEL_FACTORIES[model_key](**kwargs), extra


def run_job(job: Tuple[str, str, int, int, str]) -> Dict[str, Any]:
    dataset, model_key, split_seed, fold, out, dev = job
    from common import default_config, load_bundle, revision, gcv
    from multiclass_eventwise_runner import context_rows, loader
    torch.set_num_threads(1)
    if dataset not in _BUNDLES:
        _BUNDLES[dataset] = load_bundle(dataset)
    bundle = _BUNDLES[dataset]
    config = default_config()
    key = (dataset, split_seed)
    if key not in _FOLDS:
        _FOLDS[key] = gcv.build_folds(bundle, config.seq_len, 5, 7, split_seed)[0]
    parts = _FOLDS[key][fold]
    # The submitted split keeps the submitted model seeds (13 * 100 + fold).
    model_seed = 13 * 100 + fold + (0 if split_seed == 202604 else 10_000 * split_seed)
    fitted = context_rows(parts["train"], config.seq_len)
    reduced, _ = revision.prune_features_from_training_rows(bundle, fitted)
    prep = revision.TrainOnlyPreprocessor(config.clip_quantile_low, config.clip_quantile_high)
    prep.fit(reduced.features[fitted], reduced.labels[fitted])
    x = prep.transform(reduced.features)
    mb = revision.with_normal_deviation_schema(copy.copy(reduced), x.shape[1])
    tl = loader(revision, x, bundle.labels, parts["train"], config, True, model_seed)
    vl = loader(revision, x, bundle.labels, parts["validation"], config, False, model_seed)
    sl = loader(revision, x, bundle.labels, parts["test"], config, False, model_seed)
    seed_all(model_seed)
    name, model, flags = build(model_key, mb, config)
    start = time.perf_counter()
    info = train(name, model, tl, vl, config, flags, len(bundle.class_names))
    seconds = time.perf_counter() - start
    device = torch.device("cpu")
    yv, pv, _, _ = revision.predict_loader(model, vl, device)
    labels = np.arange(len(bundle.class_names))
    if dev:
        # Development mode: the test fold is never predicted or scored.
        pv_, rv_, fv_, _ = precision_recall_fscore_support(yv, pv, labels=labels, zero_division=0)
        row = {"dataset": dataset, "model": model_key, "split_seed": split_seed, "fold": fold,
               "val_macro_f1": float(f1_score(yv, pv, average="macro", zero_division=0)),
               "params": int(sum(q.numel() for q in model.parameters())),
               "train_seconds": seconds, "best_epoch": info["best_epoch"],
               "epochs_executed": info["epochs_executed"]}
        for i, c in enumerate(bundle.class_names):
            row[f"val_f1_{c}"] = float(fv_[i])
        return row
    yt, pt, prob, _ = revision.predict_loader(model, sl, device)
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=labels, zero_division=0)
    pred_dir = Path(out) / "predictions" / dataset / model_key.replace("/", "_")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": parts["test"], "truth": yt, "pred": pt}).to_csv(
        pred_dir / f"split{split_seed}_fold{fold}.csv", index=False)
    row = {
        "dataset": dataset, "model": model_key, "split_seed": split_seed, "fold": fold,
        "val_macro_f1": float(f1_score(yv, pv, average="macro", zero_division=0)),
        "test_macro_f1": float(f.mean()), "test_accuracy": float(accuracy_score(yt, pt)),
        "test_macro_p": float(p.mean()), "test_macro_r": float(r.mean()),
        "params": int(sum(q.numel() for q in model.parameters())),
        "train_seconds": seconds, "best_epoch": info["best_epoch"],
        "epochs_executed": info["epochs_executed"],
    }
    for i, c in enumerate(bundle.class_names):
        row[f"f1_{c}"] = float(f[i])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--datasets", default="IGCPS,WDT,ICS-Flow")
    ap.add_argument("--splits", default="202604,1,2,3,4")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--dev", action="store_true",
                    help="validation-only development run; test folds are not evaluated")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result_file = a.out / "fold_rows.csv"
    done = set()
    if result_file.exists():
        prev = pd.read_csv(result_file)
        done = set(zip(prev.dataset, prev.model, prev.split_seed, prev.fold))
    order = {"ICS-Flow": 0, "WDT": 1, "IGCPS": 2}  # longest first
    jobs = [
        (d, m, int(s), k, str(a.out), a.dev)
        for d in a.datasets.split(",") for m in a.models.split(",")
        for s in a.splits.split(",") for k in range(5)
        if (d, m, int(s), k) not in done
    ]
    jobs.sort(key=lambda j: order.get(j[0], 3))
    print(f"{len(jobs)} jobs, {len(done)} already done", flush=True)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=40) as pool:
        for n, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
            pd.DataFrame([row]).to_csv(result_file, mode="a", index=False,
                                       header=not result_file.exists())
            test = f" test={row['test_macro_f1']:.4f}" if "test_macro_f1" in row else ""
            print(f"[{n}/{len(jobs)}] {row['dataset']} {row['model']} s={row['split_seed']} "
                  f"f={row['fold']} val={row['val_macro_f1']:.4f}{test} "
                  f"({row['train_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
