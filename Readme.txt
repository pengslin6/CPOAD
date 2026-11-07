CPOAD source code and public fused datasets
===========================================

Repository: https://github.com/pengslin6/CPOAD

This snapshot publishes the author's current CPOAD implementation without
changing the supplied Python source files. CPOAD is the manuscript name;
the implementation and command-line model selector are named CPCD5.
CAFOAD_major_revision.py is retained as a shared implementation dependency,
not as the current CPOAD model definition.

Current model and entry points
-------------------------------

revision3/rev3_cpcd5.py
  CPOAD (CPCD5): cross-domain and intra-physical predictive consistency,
  residual-calibrated coupling loss, memory-based regularization, and variants.

revision3/rev3_cpcd3.py
  Contains split_schema, which is directly imported by the CPOAD model.

revision3/common.py
  Loads the shared implementation and points dataset paths to the repository
  root. Keep the revision3/ directory structure unchanged.

revision3/rev3_cv.py
  Purged grouped cross-validation and retrained-variant entry point. It can
  perform validation-only development with --dev; without --dev it also
  evaluates held-out test folds.

revision3/rev3_online_runner.py
  Three-dataset prequential replay, guarded and ungated updates, and bounded
  drift/poisoning scenarios. Select --model CPCD5 explicitly: the unchanged
  source file's default is CPCD4, not CPOAD.

Required shared dependencies
-----------------------------

CAFOAD_major_revision.py
multiclass_eventwise_runner.py
multiclass_grouped_cv_runner.py
revision3/rev3_models.py
revision3/rev3_cpcd.py
revision3/rev3_cpcd4.py

All six files are retained because the current entry points import them.
The older model families are compatibility dependencies, not renamed CPOAD
models. Additional optional baseline modules are not part of this snapshot;
only request model keys whose implementations are present.

Environment
------------

Use Python 3.10 or newer with NumPy, pandas, SciPy, scikit-learn, PyTorch,
and psutil. The supplied runners use CPU execution. No source modification
or new training was performed as part of publishing this snapshot.

Public and restricted data
---------------------------

The existing public fused datasets are preserved unchanged at the root:

  wdt_final_unified.csv
  ics_flow_final_unified.csv

The source expects these names through revision3/common.py. IGCPS is not
included: its underlying records are proprietary to Shenzhen Gas Corporation
and release requires authorization. Authorized users can place
igcps_final_unified.csv at the repository root and add IGCPS to --datasets.
No private data, model checkpoints, local experiment results, or manuscript
editing scripts are included in this update.

Example commands
------------------

Run these commands from the repository root. These are entry-point examples,
not a claim that every manuscript table has been independently reproduced.

Install dependencies:
  python -m pip install numpy pandas scipy scikit-learn torch psutil

Check the entry-point options:
  python revision3/rev3_cv.py --help
  python revision3/rev3_online_runner.py --help

Five-fold CPOAD evaluation on the public datasets:
  python revision3/rev3_cv.py --out results/cpoad_cv --models CPCD5 --datasets WDT,ICS-Flow --splits 202604 --workers 1

Retrained CPOAD component variants on the same split assignment:
  python revision3/rev3_cv.py --out results/cpoad_ablation --models CPCD5,CPCD5-w/o-rccl,CPCD5-w/o-contrastive,CPCD5-w/o-intra,CPCD5-w/o-cross,CPCD5-w/o-consistency --datasets WDT,ICS-Flow --splits 202604 --workers 1

Online replay on the public datasets:
  python revision3/rev3_online_runner.py --out results/cpoad_online --model CPCD5 --datasets WDT,ICS-Flow --seed 13 --fold 0

Use a fresh output directory for a different model, configuration, or
development/test mode. Increase --workers only when resources permit.
Repeated split assignments are not independent acquisition sessions.

Removed legacy entry points
---------------------------

The superseded combinenew.py, compute_efficiency_metrics.py,
igcps_multiclass_online_runner.py, and run_grouped_decision_fusion.py have
been removed from the current tree at the author's request. Earlier versions
remain recoverable from repository history; history has not been rewritten.
