#!/bin/bash
set -e
cd /Users/timrobarge/CSE5820/project
source poker-rl/bin/activate
export PYTHONUNBUFFERED=1

FLAGS="--reuse-blueprint --skip-ablation --tuned-only"

echo "================================================================"
echo " PHASE 1: Residual Mix Weight Sweep (balanced mix)"
echo "   Running residual_030 and residual_045 in parallel"
echo "================================================================"

python -m hulhe_bot.cli run_experiments --config configs/residual_030.json $FLAGS \
  > artifacts/residual_030.log 2>&1 &
PID1=$!

python -m hulhe_bot.cli run_experiments --config configs/residual_045.json $FLAGS \
  > artifacts/residual_045.log 2>&1 &
PID2=$!

echo "  residual_030 PID=$PID1"
echo "  residual_045 PID=$PID2"
echo "  Waiting for Phase 1..."
wait $PID1 && echo "  residual_030 DONE" || echo "  residual_030 FAILED (exit $?)"
wait $PID2 && echo "  residual_045 DONE" || echo "  residual_045 FAILED (exit $?)"

echo ""
echo "================================================================"
echo " PHASE 2: TAG-Heavy Opponent Mix (60% TAG)"
echo "   Running tag_heavy_015, tag_heavy_030, tag_heavy_045 in parallel"
echo "================================================================"

python -m hulhe_bot.cli run_experiments --config configs/tag_heavy_015.json $FLAGS \
  > artifacts/tag_heavy_015.log 2>&1 &
PID3=$!

python -m hulhe_bot.cli run_experiments --config configs/tag_heavy_030.json $FLAGS \
  > artifacts/tag_heavy_030.log 2>&1 &
PID4=$!

python -m hulhe_bot.cli run_experiments --config configs/tag_heavy_045.json $FLAGS \
  > artifacts/tag_heavy_045.log 2>&1 &
PID5=$!

echo "  tag_heavy_015 PID=$PID3"
echo "  tag_heavy_030 PID=$PID4"
echo "  tag_heavy_045 PID=$PID5"
echo "  Waiting for Phase 2..."
wait $PID3 && echo "  tag_heavy_015 DONE" || echo "  tag_heavy_015 FAILED (exit $?)"
wait $PID4 && echo "  tag_heavy_030 DONE" || echo "  tag_heavy_030 FAILED (exit $?)"
wait $PID5 && echo "  tag_heavy_045 DONE" || echo "  tag_heavy_045 FAILED (exit $?)"

echo ""
echo "================================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo "Reports:"
echo "  artifacts/residual_030_report.json"
echo "  artifacts/residual_045_report.json"
echo "  artifacts/tag_heavy_015_report.json"
echo "  artifacts/tag_heavy_030_report.json"
echo "  artifacts/tag_heavy_045_report.json"
echo ""
echo "Logs:"
echo "  artifacts/residual_030.log"
echo "  artifacts/residual_045.log"
echo "  artifacts/tag_heavy_015.log"
echo "  artifacts/tag_heavy_030.log"
echo "  artifacts/tag_heavy_045.log"
