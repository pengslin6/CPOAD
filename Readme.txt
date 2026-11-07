CPOAD: single-file implementation and public fused datasets
===========================================================

Repository: https://github.com/pengslin6/CPOAD

cpoad.py is the only Python source file in the current repository snapshot.
It is uploaded unchanged from the author's supplied file. The previous
multi-file model and runner implementation has been removed from the current
tree at the author's request. Those versions remain recoverable from Git
history; no history has been rewritten.

Included files
---------------

cpoad.py
  Contains data loading, train-only preprocessing, purged grouped folds,
  the predictive-consistency detector and its losses, offline training,
  guarded online updates, streaming comparison and poisoning experiments,
  and confusion-matrix/online-trace figure generation.

wdt_final_unified.csv
ics_flow_final_unified.csv
  The existing two public fused datasets, preserved unchanged.

IGCPS is not uploaded. Its underlying records are proprietary to Shenzhen
Gas Corporation, and release requires authorization. Authorized users may
provide an IGCPS fused CSV through --data.

Dependencies and verification scope
------------------------------------

Python with NumPy, pandas, scikit-learn, and PyTorch. Plot generation also
requires matplotlib. Startup checks for this upload use Python 3.12.7 and
PyTorch 2.7.0+cpu on a Windows workstation. Other Python/PyTorch environments,
Jetson execution, GPU acceleration, and all manuscript numerical results
require separate validation. Publishing this file does not establish those
results. The implementation itself has not been modified during upload.

Install the dependencies in an appropriate environment:
  python -m pip install numpy pandas scikit-learn torch matplotlib

Data paths
------------

The supplied source was originally kept inside revision3/ and its default
data directory is the parent of the directory containing the script.
To preserve that source unchanged while publishing cpoad.py at the repository
root, always pass --data explicitly, as in every example below. No old local
code modules are needed.

Command examples
------------------

Run commands from the repository root. Start with --workers 1 on devices with
limited memory. These examples invoke the supplied implementation; they are
not a guarantee of equality with every manuscript result.

Show the command options (does not start training):
  python cpoad.py --help

Train one WDT fold and save its weights:
  python cpoad.py train WDT 0 --data wdt_final_unified.csv --save wdt_f0.pt --out results/train_WDT

Run the simple online entry point from saved weights:
  python cpoad.py online WDT 0 --data wdt_final_unified.csv --ckpt wdt_f0.pt --out results/online_WDT

Generate the online-comparison table over all five folds:
  python cpoad.py online-table WDT --all --workers 1 --data wdt_final_unified.csv --ckpt-dir checkpoints --out results/online_table_WDT

Run poisoning experiments on one fold:
  python cpoad.py poison WDT 0 --workers 1 --data wdt_final_unified.csv --ckpt-dir checkpoints --out results/poison_WDT

Generate a five-fold pooled confusion matrix:
  python cpoad.py confusion ICS-Flow --workers 1 --data ics_flow_final_unified.csv --ckpt-dir checkpoints --out results/confusion_ICS

Generate fold-0 online traces and figures:
  python cpoad.py fig6 WDT --data wdt_final_unified.csv --ckpt-dir checkpoints --out results/fig6_WDT

Use the matching dataset name and --data CSV path for other datasets.
--ckpt-dir reuses existing checkpoints when present, so use a fresh directory
for a different source revision, data schema, or training configuration.
Private data, checkpoints, local result files, and the manuscript are not
included in this update.
