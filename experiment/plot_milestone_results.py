"""Create compact PNG figures and measured summaries for the four-method report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run_milestone_experiments import BUDGETS, RESULTS, SEEDS
from .run_single_attacker import MODEL_LABELS, MODEL_ORDER, curve


OUTPUT = Path(__file__).resolve().parent / "ignore/mid"


def load_case(count):
    runs = []
    for seed in SEEDS:
        for budget in BUDGETS:
            path = RESULTS / f"attackers{count}/seed{seed}_budget{budget:g}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["config"]["attackers"] == count
            runs.append(payload)
    assert len(runs) == 21
    per_seed = {}
    for seed in SEEDS:
        for model in MODEL_ORDER:
            points = [(r["average_communication_cost"], r["integrity_success_rate"])
                      for run in runs if run["config"]["seed"] == seed
                      for r in run["results"] if r["model"] == model]
            assert len(points) == 7
            per_seed[(seed, model)] = curve(points)
    low = max(xs.min() for xs, _ in per_seed.values())
    high = min(xs.max() for xs, _ in per_seed.values())
    assert low < high
    target = 1.0 if low <= 1.0 <= high else (low + high) / 2
    grid = np.unique(np.r_[np.linspace(low, high, 240), target])
    curves = {}
    matched = []
    for model in MODEL_ORDER:
        ys = np.array([np.interp(grid, *per_seed[(seed, model)]) for seed in SEEDS])
        mean = ys.mean(axis=0)
        ci = 1.96 * ys.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        index = int(np.where(grid == target)[0][0])
        curves[model] = (mean, ci)
        matched.append(dict(attacker_count=count, model=model, communication_cost=target,
                            integrity_success_rate=float(mean[index]), ci95=float(ci[index])))
    return runs, grid, curves, matched


def plot_case(count, grid, curves, matched, name):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    target = matched[0]["communication_cost"]
    best_ml = max((r for r in matched if r["model"].startswith("ml_")), key=lambda r:r["integrity_success_rate"])["model"]
    colors = ("#1f77b4", "#ff7f0e", "#218739", "#d62728")
    lower, upper = [], []
    for model, color in zip(MODEL_ORDER, colors):
        mean, ci = curves[model]
        bold = model in (best_ml, "q_learning_rl")
        ax.plot(grid, mean, color=color, label=MODEL_LABELS[model],
                linewidth=2.6 if bold else 1.6)
        ax.fill_between(grid, np.clip(mean-ci,0,1), np.clip(mean+ci,0,1),
                        color=color, alpha=0.12, linewidth=0)
        row = next(r for r in matched if r["model"] == model)
        ax.scatter([target], [row["integrity_success_rate"]], color=color, s=25, edgecolors="white", linewidths=0.6, zorder=5)
        lower.append(float(np.clip(mean-ci,0,1).min()))
        upper.append(float(np.clip(mean+ci,0,1).max()))
    ax.axvline(target, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Average normalized communication cost", labelpad=7)
    ax.set_ylabel("Integrity success rate", labelpad=7)
    ax.set_title(f"{'Single attacker' if count==1 else 'Three staggered attackers'}: success-cost curves", pad=10)
    span = max(upper)-min(lower)
    ax.set_ylim(max(0,min(lower)-0.04*span), min(1,max(upper)+0.04*span))
    ax.set_xlim(grid[0],grid[-1])
    ax.grid(alpha=0.23)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", framealpha=0.94, fontsize=9)
    for text, model in zip(legend.get_texts(),MODEL_ORDER):
        if model in (best_ml,"q_learning_rl"):
            text.set_fontweight("bold")
    fig.text(0.5,0.022,"Mean and approximate 95% CI over 3 seeds; dashed line marks matched communication cost.",
             ha="center",fontsize=8,color="#555555")
    fig.subplots_adjust(left=0.10,right=0.98,top=0.90,bottom=0.17)
    fig.savefig(OUTPUT/name,dpi=240,facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summary, raw, curve_rows = [], [], []
    for count,name in [(1,"01_single_attacker_success_cost.png"),(3,"02_multiple_attackers_success_cost.png")]:
        runs,grid,curves,matched = load_case(count)
        plot_case(count,grid,curves,matched,name)
        summary.extend(matched)
        for run in runs:
            for result in run["results"]:
                raw.append(dict(attacker_count=count,seed=run["config"]["seed"],
                                nominal_budget=run["config"]["budget"],**result))
        for model,(mean,ci) in curves.items():
            for x,y,h in zip(grid,mean,ci):
                curve_rows.append(dict(attacker_count=count,model=model,communication_cost=x,
                                       integrity_success_rate=y,ci95=h))
        print(count,"attackers",json.dumps(matched),flush=True)
    for filename,rows in [("matched_cost_summary.csv",summary),("raw_results.csv",raw),("curve_data.csv",curve_rows)]:
        with (RESULTS/filename).open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
