#!/bin/bash
set -e
cd /Users/timrobarge/CSE5820/project
source poker-rl/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Run 1: residual_mix_weight=0.30 + subgame ==="
python -m hulhe_bot.cli run_experiments --config configs/residual_030.json --reuse-blueprint

echo ""
echo "=== Run 2: residual_mix_weight=0.45 + subgame ==="
python -m hulhe_bot.cli run_experiments --config configs/residual_045.json --reuse-blueprint

echo ""
echo "=== Both residual weight experiments complete ==="
