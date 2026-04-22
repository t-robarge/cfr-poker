import json, os

reports = {}
for f in os.listdir("artifacts"):
    if f.endswith("_report.json"):
        path = f"artifacts/{f}"
        try:
            reports[f] = json.load(open(path))
        except:
            pass

for name, data in sorted(reports.items()):
    print(f"\n{'='*60}")
    print(f"FILE: {name}")
    print("="*60)

    if "evaluations" in data:
        for policy, opps in data["evaluations"].items():
            for opp, vals in opps.items():
                mean = vals.get("sb_per_100_mean", "?")
                ci = vals.get("sb_per_100_ci95", "?")
                print(f"  {policy} vs {opp}: mean={mean}, ci95={ci}")

    if "head_to_head" in data:
        for matchup, vals in data["head_to_head"].items():
            mean = vals.get("sb_per_100_mean", "?")
            ci = vals.get("sb_per_100_ci95", "?")
            print(f"  H2H {matchup}: mean={mean}, ci95={ci}")

    cfg = data.get("config", {})
    print(f"  flop_buckets={cfg.get('flop_buckets','?')}, turn_buckets={cfg.get('turn_buckets','?')}, river_buckets={cfg.get('river_buckets','?')}")
    print(f"  rollout={cfg.get('flop_rollout_samples','?')}, abstraction_samples={cfg.get('abstraction_samples','?')}")
    print(f"  training_iters={cfg.get('training_iterations','?')}, fine_tune_hands={cfg.get('fine_tune_hands','?')}")
    print(f"  eval_hands={cfg.get('default_eval_hands','?')}, eval_seeds={cfg.get('default_eval_seeds','?')}")
