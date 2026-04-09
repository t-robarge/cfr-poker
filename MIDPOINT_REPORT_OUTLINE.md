# Midpoint Report Outline: Game-Theoretic Poker Bot (4 Pages)

---

## PAGE 1: INTRODUCTION & PROBLEM STATEMENT

### 1.1 Executive Summary (½ page)
- **Project Goal**: Develop a competitive Heads-Up Limit Texas Hold'em (HULHE) bot using game-theoretic algorithms
- **Core Challenge**: Solving large imperfect-information games without exhaustive enumeration
- **Approach**: Game abstraction → Counterfactual Regret Minimization (CFR) → Reinforcement Learning refinement
- **Honest Status**: Pipeline is fully functional. Bot beats random and mirror opponents convincingly. Major open problem: bot gets exploited by tight-aggressive play (−75 to −80 sb/100), which we diagnose and plan to fix.

### 1.2 Motivation & Context (¾ page)
- Why poker is a testbed for AI: imperfect information, stochasticity, strategic complexity
- Game-theoretic approach advantages over RL-only methods:
  - Convergence guarantees to Nash equilibrium
  - Interpretable strategies (policy table is human-readable)
  - Data-efficient compared to pure self-play
- Industry precedent: Libratus (2017) and Pluribus (2019) used abstraction + CFR to solve poker variants
- Applications beyond poker: negotiation, security, auctions

### 1.3 Problem Formulation (½ page)
- **Key Challenge**: Full HULHE game tree is too large to solve directly
- **Solution Architecture** (three-stage pipeline):
  1. **Abstraction**: Map the ~10^162 game states into manageable equity buckets
  2. **Blueprint Training**: Run Monte Carlo CFR on the abstract game to find near-equilibrium strategy
  3. **Fine-Tuning**: Layer residual RL adjustments to adapt to specific opponents

---

## PAGE 2: METHODOLOGY & SYSTEM ARCHITECTURE

### 2.1 Game Abstraction Layer (¾ page)
- **Bucketing Strategy**: Group hands by equity (win probability) + hand features
  - Preflop: 169 buckets (canonical starting hand classes, e.g., AA, AKs, AKo, ...)
  - Flop: 12 equity buckets × feature buckets (made hand class × draw class = up to 15 features per equity bucket)
  - Turn: 12 equity buckets × feature buckets (same structure)
  - River: 10 equity buckets × made hand class (exact showdown eval, no rollout needed)
  
- **Equity Estimation**:
  - Preflop/Flop/Turn: Monte Carlo rollout with `flop_rollout_samples` (currently 4) and `turn_rollout_samples` (currently 4)
  - River: Exact enumeration over all remaining opponent holdings
  - Canonical suit isomorphism (e.g., Ah-Kh ≡ As-Ks) to reduce redundancy
  - LRU cache (200k entries) for repeated lookups

- **Configuration Issue** (see §4.1): Current rollout samples (4 per street) are very low, creating noisy equity estimates. Increasing to 16-32 is expected to improve bucket accuracy significantly.

### 2.2 Blueprint Training: Monte Carlo CFR with External Sampling (¾ page)
- **Algorithm**: MCCFR with external sampling and stochastically-weighted averaging
  - Recursive depth-first traversal of abstract game tree
  - Each infoset maintains two arrays: **accumulated regrets** and **accumulated strategy sums**
  - For the traversing player: explore ALL actions, compute counterfactual values, update regrets
  - For opponent: SAMPLE one action proportional to current strategy (external sampling — no reach probability tracking needed)
  
- **Regret Matching**:
  - Current strategy at any infoset: actions proportional to positive accumulated regrets
  - `σ(a) = max(0, R(a)) / Σ max(0, R(a'))` for all actions a'
  - If all regrets ≤ 0, play uniform random
  
- **Convergence**: 
  - Individual infoset strategies do NOT converge — but the TIME-AVERAGED strategy converges to Nash equilibrium
  - Final policy = normalized strategy sums over all iterations
  - Theoretical bound: O(1/√T) exploitability

- **Training Runs Completed**:
  - 25,000 iterations (mccfr_25k config)
  - 100,000 iterations (mccfr_100k config)
  - Both used 120 abstraction samples, 4 rollout samples per street

### 2.3 Refinement: Residual RL Fine-Tuning (½ page)
- **Goal**: Improve blueprint against non-equilibrium opponents
- **Method**: Q-learning on a residual table overlaid on the blueprint
  - Play episodes against opponent mix (50% mirror, 25% loose-passive, 25% tight-aggressive)
  - Update per-action Q-values: `Q(s,a) ← Q(s,a) + α(r − Q(s,a))`, α = 0.15
  - Final policy = (1 − w) × blueprint + w × softmax(Q-values), w = 0.15, temperature = 0.5
  - Evaluate every `fine_tune_eval_interval` episodes; keep best residual by validation score

- **Current Status**: Fine-tuning has not produced measurable improvement over blueprint (see §3.2, §4.1)

---

## PAGE 3: PROGRESS & CURRENT STATUS

### 3.1 Completed Work (¾ page)

**Infrastructure & Core Implementation** ✓
- Full HULHE game engine: limit betting structure, 4-bet cap, blind posting, showdown resolution
- Abstraction pipeline: hand evaluation → Monte Carlo equity → bucketing → abstract game tree
- CFR trainer: external sampling with MCCFR, CFR+, and DCFR algorithm variants
- Policy runtime: action sampling from policy table with optional residual mixing
- Baseline agents: RandomAgent, LoosePassiveAgent, TightAggressiveAgent, PolicyAgent (mirror)

**Experimental Framework** ✓
- Evaluation system: heads-up matches measured in sb/100 (small bets per 100 hands)
- Multi-seed evaluation with 95% confidence intervals and win/loss/uncertain verdicts
- Solver validation on Kuhn Poker (known equilibrium) via LiteEFG + OpenSpiel integration
- Progress checkpoints: training iteration logs, match progress (every 10%)
- CLI with subcommands: `build_abstract_game`, `train_blueprint`, `fine_tune`, `evaluate`, `run_experiments`

**Code Quality** ✓
- Python 3.12, fully type-hinted dataclasses, frozen immutable game state
- Unit tests for engine, bucketing, abstract game, experiments
- Deterministic via seeded RNG (reproducible results)
- JSON artifact format for policies and reports

### 3.2 Actual Experimental Results (¾ page)

**Table 1: Blueprint vs Opponents (MCCFR, 100k iterations, 300 hands × 3 seeds)**

| Opponent          | sb/100 (mean) | ±CI (95%) | Verdict     |
|-------------------|---------------|-----------|-------------|
| random            | +31.61        | ±15.47    | **WIN**     |
| loose_passive     | −3.78         | ±23.41    | uncertain   |
| tight_aggressive  | −79.22        | ±17.70    | **LOSS**    |
| mirror            | +21.78        | ±25.67    | uncertain   |

**Table 2: Blueprint vs Opponents (MCCFR, 100k iterations, 500 hands × 4 seeds, rerun)**

| Opponent          | sb/100 (mean) | ±CI (95%) | Verdict     |
|-------------------|---------------|-----------|-------------|
| random            | +29.45        | ±17.69    | **WIN**     |
| loose_passive     | −12.68        | ±18.73    | uncertain   |
| tight_aggressive  | −75.40        | ±10.64    | **LOSS**    |
| mirror            | +21.18        | ±9.53     | **WIN**     |

**Table 3: Iteration Scaling (25k vs 100k, same abstraction)**

| Opponent          | 25k sb/100 | 100k sb/100 | Δ       |
|-------------------|------------|-------------|---------|
| random            | +27.89     | +31.61      | +3.72   |
| loose_passive     | −3.83      | −3.78       | +0.05   |
| tight_aggressive  | −78.28     | −79.22      | −0.94   |
| mirror            | +5.39      | +21.78      | +16.39  |

**Head-to-Head Results (first_metrics run, 50 hands × 5 seeds)**
- Blueprint vs CFR+ ablation: +6.76 ± 14.13 sb/100 (uncertain — tiny sample, 20 iteration ablation)
- Tuned vs Blueprint: +3.68 ± 14.30 sb/100 (uncertain — fine-tuning did not improve)

**Solver Validation** ✓
- Kuhn Poker: exploitability dropped from 0.500 → 0.111 in 50 iterations (improved = true)

### 3.3 Key Observations (½ page)

1. **Bot reliably beats random** (~+28-31 sb/100): Core CFR logic is working correctly
2. **Bot beats its own mirror** (~+21 sb/100 with 100k iters): Position-awareness and seat-alternation working
3. **More iterations help for mirror** (+5.39 → +21.78 from 25k→100k) but NOT for tight-aggressive (−78 → −79)
4. **Catastrophic loss to tight-aggressive** (−75 to −80 sb/100): This is the dominant problem — see §4.1
5. **Fine-tuning shows no improvement**: Residual RL unable to overcome the fundamental abstraction weakness
6. **Wide confidence intervals**: Sample sizes too small (300-500 hands) for definitive conclusions on many matchups
7. **CFR+ vs MCCFR comparison is inconclusive**: Ablation used only 20 iterations — not a real comparison

---

## PAGE 4: CHALLENGES, DIAGNOSIS & PATH FORWARD

### 4.1 Root Cause Analysis: Why the Bot Loses to Tight-Aggressive (¾ page)

This is the project's central problem. The TightAggressiveAgent uses bucket_percentile directly:
- Folds when percentile < 0.28 (preflop) or < 0.35 (postflop)
- Raises when percentile ≥ 0.72 (preflop) or ≥ 0.75 (postflop)  
- Otherwise calls

**Diagnosis: Three compounding issues**

**Issue 1: Coarse Equity Estimation (Primary)**
- Current config uses only 4 Monte Carlo rollout samples per street
- With 4 samples, equity estimates have enormous variance — a hand with true equity 0.55 might be bucketed as 0.25 or 0.80
- This means the abstract game the bot trains on has scrambled hand-strength information
- The tight-aggressive agent reads equity perfectly (it uses `bucket_percentile` at runtime), while our CFR policy was trained on noisy equity
- **Fix**: Increase `flop_rollout_samples` and `turn_rollout_samples` to 32-64. Increase `abstraction_samples` from 120 to 2000+.

**Issue 2: Abstraction Samples Too Low (Secondary)**
- Current: 120 `abstraction_samples` and 120 `river_payoff_samples` to build the entire abstract game
- These samples define the bucket transition probabilities and showdown equity tables
- With so few samples, the abstract game is a poor approximation of real poker
- **Fix**: Increase `abstraction_samples` to 2000+ and `river_payoff_samples` to 2000+.

**Issue 3: Fine-Tuning is Underpowered (Tertiary)**
- Only 200 fine-tune hands with 100 validation hands — far too few to learn meaningful Q-values
- Evaluation interval of 100 means only 2 checkpoints total
- Even default config (100k hands, 2k validation) may be insufficient given the abstraction quality issues
- **Fix**: First fix abstraction quality, then fine-tune with 50k+ hands against a TAG-heavy opponent mix.

**Why more CFR iterations don't help**: The 25k→100k comparison shows tight-aggressive performance is FLAT (−78 → −79). The problem is not convergence — it's that the bot is converging to the optimal strategy *for the wrong game* (the noisy abstraction).

### 4.2 Additional Challenges (½ page)

**Challenge: Computational Cost**
- 100k CFR iterations took significant time; evaluation adds hours
- Added progress checkpoints (every 10k iterations, every 10% of match hands) for visibility
- Resolution: Accept runtime; focus on making each run count with better parameters

**Challenge: Statistical Significance**
- Most results have wide CIs (±10-25 sb/100) due to 300-500 hand samples
- Some "uncertain" verdicts may hide real effects  
- Resolution: Run proper evaluation with 2000+ hands × 5+ seeds

**Challenge: No Controlled CFR+ vs MCCFR Comparison**
- The only ablation used 20 iterations — effectively meaningless
- Resolution: Run both algorithms at same iteration count with proper evaluation

### 4.3 Remediation Plan (½ page)

**Phase 1: Fix Abstraction Quality (Week of April 7)**
- Increase rollout samples: 4 → 32 (flop), 4 → 32 (turn)
- Increase abstraction samples: 120 → 2000
- Increase river payoff samples: 120 → 2000
- Rebuild abstract game and retrain blueprint (100k iterations)

**Phase 2: Proper Evaluation (Week of April 14)**
- Evaluate with 2000 hands × 5 seeds per opponent (statistically meaningful)
- Run controlled ablation: MCCFR vs CFR+ at 100k iterations, same abstraction
- Run iteration scaling study: 25k / 50k / 100k / 200k iterations

**Phase 3: Fine-Tuning at Scale (Week of April 21)**
- With improved abstraction, retrain residual RL: 50,000 hands
- Test opponent mix variations (more TAG weight, since that's our weakness)
- Validate improvement with head-to-head vs blueprint

**Phase 4: Final Report & Analysis (Week of April 28)**
- Complete ablation tables
- Convergence analysis (exploitability vs iterations)
- Strategy interpretation (what does the bot actually do preflop?)

### 4.4 Success Criteria & Current Status

| Criterion                          | Target       | Current       | Status        |
|------------------------------------|--------------|---------------|---------------|
| Beat random agent                  | +20 sb/100   | +29-32 sb/100 | ✓ MET         |
| Beat mirror (self-play)            | ≥0 sb/100    | +21 sb/100    | ✓ MET         |
| Beat tight-aggressive              | ≥0 sb/100    | −75 to −80    | ✗ FAILING     |
| Beat loose-passive                 | ≥0 sb/100    | −4 to −13     | ⚠ UNCERTAIN  |
| Kuhn solver test                   | converges    | 0.50→0.11     | ✓ MET         |
| CFR+ vs MCCFR comparison          | controlled   | not done      | ⚠ PENDING    |
| Fine-tuning improves blueprint     | +5 sb/100    | +3.68±14.30   | ✗ NOT YET     |
| Statistical confidence             | CI < ±10     | ±10 to ±25    | ⚠ NEEDS WORK |

---

## VISUAL ELEMENTS TO INCLUDE

1. **Figure 1**: System architecture diagram (abstraction → CFR → RL → evaluation pipeline)
2. **Table 1-3**: Performance results (already outlined above — use real data only)
3. **Figure 2**: Iteration scaling chart (25k vs 100k sb/100 by opponent)
4. **Table 4**: Success criteria matrix (§4.4)
5. **Figure 3**: Planned convergence curves from Phase 2 evaluation
