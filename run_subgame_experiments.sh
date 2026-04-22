#!/bin/bash
set -e
cd /Users/timrobarge/CSE5820/project
source poker-rl/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Run A: Tabular + Subgame ==="
python -m hulhe_bot.cli run_experiments --config configs/subgame_tabular.json --reuse-blueprint

echo ""
echo "=== Run B: NN + Subgame ==="
python -m hulhe_bot.cli run_experiments --config configs/subgame_nn.json --reuse-blueprint

echo ""
echo "=== Both experiments complete ==="
