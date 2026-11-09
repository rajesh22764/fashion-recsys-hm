#!/usr/bin/env bash
# Reproduce the leak-free pipeline (k-core computed on the training period only).
#   PY=/path/to/python bash scripts/run_pipeline.sh          # main evaluation window (shift 0)
#   PY=/path/to/python bash scripts/run_pipeline.sh 2        # same, window moved 2 weeks earlier (no two-tower)
# torch and LightGBM crash together on macOS, so candidate building and ranker training are separate processes.
set -euo pipefail
PY=${PY:-python}
SHIFT=${1:-0}
D=data/hm
FLAGS="--kcore train --shift-weeks $SHIFT"

if [ "$SHIFT" = "0" ]; then
  $PY -u -m src.run_hm "$D" $FLAGS                                   # tuned baselines
  $PY -u -m src.run_twotower "$D" $FLAGS                             # two-tower model
  $PY -u -m src.run_ranker "$D" $FLAGS --stage candidates --twotower
  $PY -u -m src.run_ranker "$D" $FLAGS --stage train --twotower
fi
$PY -u -m src.run_ranker "$D" $FLAGS --stage candidates
$PY -u -m src.run_ranker "$D" $FLAGS --stage train
if [ "$SHIFT" = "0" ]; then
  $PY -u -m src.analyze "$D" $FLAGS --twotower                       # CIs, fusion baseline, ablations
  $PY -u -m src.cold_start "$D" $FLAGS                               # cold-start experiment
fi
