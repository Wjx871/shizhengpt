#!/usr/bin/env bash
set -euo pipefail
cd /root/private_data/shizhengpt-evaluation
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/verify_model.py > evaluation/outputs/integrity.log 2>&1
.venv/bin/python scripts/smoke_test.py > evaluation/outputs/smoke.log 2>&1
.venv/bin/python scripts/batch_evaluate.py > evaluation/outputs/batch.log 2>&1
.venv/bin/python scripts/benchmark.py > evaluation/outputs/benchmark.log 2>&1

.venv/bin/python scripts/batch_evaluate.py --manifest evaluation/prompt_controls_manifest.json --output evaluation/results/prompt_controls.csv > evaluation/outputs/prompt_controls.log 2>&1
.venv/bin/python scripts/summarize_results.py > evaluation/outputs/summary.log 2>&1

.venv/bin/python scripts/batch_evaluate.py --manifest evaluation/mc_permutation_manifest.json --output evaluation/results/mc_permutations.csv > evaluation/outputs/mc_permutations.log 2>&1

.venv/bin/python scripts/summarize_controls.py > evaluation/outputs/controls_summary.log 2>&1
