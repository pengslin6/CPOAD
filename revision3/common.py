"""Shared helpers for the third-round (post-rejection) experiments.

The original runners are imported unchanged; only the dataset paths are
redirected to the fused CSVs that live in the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import multiclass_eventwise_runner as mer  # noqa: E402
import multiclass_grouped_cv_runner as gcv  # noqa: E402

DATASETS = {
    "IGCPS": ROOT / "igcps_final_unified.csv",
    "WDT": ROOT / "wdt_final_unified.csv",
    "ICS-Flow": ROOT / "ics_flow_final_unified.csv",
}
mer.DATASETS.update(DATASETS)
gcv.DATASETS.update(DATASETS)

revision = mer.load_revision()
revision.DEFAULT_DATASETS.update(DATASETS)


def load_bundle(name: str):
    return mer.original_labels(revision.load_dataset(name, DATASETS[name]))


def default_config(**overrides):
    values = dict(
        seq_len=4,
        epochs=30,
        patience=7,
        batch_size=256,
        learning_rate=8e-4,
        weight_decay=2e-4,
        max_train_windows=250_000,
        max_validation_windows=250_000,
        max_test_windows=250_000,
    )
    values.update(overrides)
    return revision.ExperimentConfig(**values)
