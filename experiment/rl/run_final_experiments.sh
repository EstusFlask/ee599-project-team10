#!/bin/bash
# Full large-scale pipeline: stage parameters from the milestone report, delayed receipts,
# price-conditioned QR-DQN (3 seeds), ablations, other RL backbones, classic ML models.
# Run from the repository root:  bash experiment/rl/run_final_experiments.sh
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1
[ -f experiment/rl/large_scale.py ] || { echo "run from the repo"; exit 1; }
PY=${PY:-/c/Users/76379/.conda/envs/ee599-rl/python.exe}
R=experiment/results/rl_large_scale_final
mkdir -p $R
TEST_SEEDS=$(seq 1000 1019)
VAL_SEEDS=$(seq 3000 3019)
PRICES16=$($PY -c "import numpy as np; print(' '.join(f'{p:.4f}' for p in np.geomspace(0.01,5,16)))")
PRICES8=$($PY -c "import numpy as np; print(' '.join(f'{p:.4f}' for p in np.geomspace(0.01,5,8)))")
TRAIN="$PY -m experiment.rl.train_large_scale --steps 600000 --price-conditioned --constraint-weight 0"
EVAL="$PY -m experiment.rl.evaluate_large_scale --delay 1"

# ---- phase A: main QR-DQN (3 seeds), tuned rule, frozen-baseline validation
for s in 2026 2027 2028; do
  $TRAIN --seeds $s --output $R/models > $R/train_qrdqn_$s.log 2>&1 &
done
$PY -m experiment.rl.tune_rule --candidates 150 --delay 1 --output $R/rule > $R/tune_rule.log 2>&1 &
$EVAL --fixed-only --seeds $VAL_SEEDS --output $R/val_fixed > /dev/null 2>&1 &
wait

# ---- phase B: ablations, other backbones, ML, rule ablations
mkdir -p $R/abl_models
$TRAIN --seeds 2026 --no-receipts --output $R/abl_raw/no_receipts > $R/train_abl_no_receipts.log 2>&1 &
$TRAIN --seeds 2026 --mask-features 0 1 2 3 4 5 6 7 8 13 --output $R/abl_raw/no_estimator > $R/train_abl_no_estimator.log 2>&1 &
$TRAIN --seeds 2026 --action-mode k --output $R/abl_raw/k_actions > $R/train_abl_k_actions.log 2>&1 &
for algo in dqn ppo a2c recurrent_ppo; do
  $TRAIN --algo $algo --seeds 2026 --output $R/backbones > $R/train_$algo.log 2>&1 &
done
$PY -m experiment.rl.train_ml_large_scale --output $R/ml/models > $R/train_ml.log 2>&1 &
$PY -m experiment.rl.run_rule_ablations --full $R/rule/gaps_receipt_delay1.csv --output $R/rule_ablation > $R/rule_ablation.log 2>&1 &
wait
for n in no_receipts no_estimator k_actions; do
  cp $R/abl_raw/$n/qrdqn_seed2026.zip $R/abl_models/qrdqn_${n}_seed2026.zip
done

# ---- phase C: evaluation on the held-out test seeds
RULES=$R/rule/chosen_receipt_delay1.json
$EVAL --rl-price-conditioned --seeds $TEST_SEEDS --prices $PRICES16 --rule-params $RULES \
    --models $R/models/qrdqn_seed2026.zip $R/models/qrdqn_seed2027.zip $R/models/qrdqn_seed2028.zip \
    --output $R/eval_main > /dev/null 2>&1 &
$EVAL --only-models --rl-price-conditioned --rl-no-receipts --seeds $TEST_SEEDS --prices $PRICES16 \
    --models $R/abl_models/qrdqn_no_receipts_seed2026.zip --output $R/eval_abl_no_receipts > /dev/null 2>&1 &
$EVAL --only-models --rl-price-conditioned --rl-mask-features 0 1 2 3 4 5 6 7 8 13 --seeds $TEST_SEEDS \
    --prices $PRICES16 --models $R/abl_models/qrdqn_no_estimator_seed2026.zip \
    --output $R/eval_abl_no_estimator > /dev/null 2>&1 &
$EVAL --only-models --rl-price-conditioned --rl-action-mode k --seeds $TEST_SEEDS --prices $PRICES16 \
    --models $R/abl_models/qrdqn_k_actions_seed2026.zip --output $R/eval_abl_k_actions > /dev/null 2>&1 &
for algo in dqn ppo a2c recurrent_ppo; do
  $EVAL --only-models --rl-price-conditioned --seeds $TEST_SEEDS --prices $PRICES16 \
      --models $R/backbones/${algo}_seed2026.zip --output $R/eval_$algo > /dev/null 2>&1 &
done
for scenario in independent waves campaign campaign_waves; do
  $EVAL --only-models --scenarios $scenario --seeds $TEST_SEEDS --prices $PRICES8 \
      --ml-models $R/ml/models/*.joblib --output $R/eval_ml_$scenario > /dev/null 2>&1 &
done
wait

# ---- merge and analyse (bootstrap over test seeds, fixed points frozen on validation seeds)
$PY - <<PYEOF
import csv, glob
R = "$R"
rows = []
files = [f"{R}/eval_main/episodes.csv"] + sorted(glob.glob(f"{R}/eval_*/episodes.csv"))
seen = set()
for f in dict.fromkeys(files):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        key = (r["scenario"], r["family"], r["policy"], r["price"], r["budget_scale"], r["seed"])
        if key not in seen:  # the same deterministic point can be evaluated in several directories
            seen.add(key)
            rows.append(r)
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(f"{R}/all_episodes.csv", "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "rows from", len(set(files)), "files")
PYEOF
$PY -m experiment.rl.analyze_large_scale --test $R/all_episodes.csv --validation $R/val_fixed/episodes.csv \
    --output $R/analysis.csv > $R/analysis.txt 2>&1
echo ALL_DONE > $R/DONE
