"""PNG-only progress figures with the strongest ML/RL at cost 1 highlighted."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import t

from .benchmark_ml_methods import ML_ORDER
from .benchmark_multi_attacker_matched_cost import POLICY_ORDER as RL_BASELINE_ORDER


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENT / "results/rl_multi_attacker_final/analysis"
CACHE = SOURCE / "ml_extension"
OUTPUT = EXPERIMENT / "ignore/all"
SEEDS = (3101, 3202, 3303)
CASES = {"single": (1,), "multi_2": (2,), "multi_3": (3,), "multi_5": (5,), "unknown": (1, 2, 3, 5)}
POLICIES = (*RL_BASELINE_ORDER, *ML_ORDER)
RL = set(RL_BASELINE_ORDER[:7])
LABELS = {
    "QR-DQN original": "QR-DQN original",
    "QR-DQN scratch multi": "QR-DQN multi-trained",
    "QR-DQN uniform fine-tuned": "QR-DQN uniform FT",
    "QR-DQN curriculum": "QR-DQN curriculum",
    "QR-DQN reliability-tuned": "QR-DQN reliability FT",
    "HistGradientBoosting": "Hist. Gradient Boosting",
}


def read_rows():
    rows = []
    for source in (SOURCE / "matched_cost_multi_raw.csv", CACHE / "ml_multi_attacker_raw.csv"):
        with source.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        for key in ("attacker_count", "seed"):
            row[key] = int(row[key])
        for key in ("mean_cost", "weighted_success_rate", "budget_scale", "gate"):
            row[key] = float(row[key])
    assert len(rows) == 20064
    assert {r["policy"] for r in rows} == set(POLICIES)
    keys = {(r["policy"], r["base_scenario"], r["attacker_count"], r["seed"], r["budget_scale"], r["gate"]) for r in rows}
    assert len(keys) == len(rows)
    return rows


def curve(raw, policy, counts):
    groups = defaultdict(list)
    for row in raw:
        if row["policy"] == policy and row["attacker_count"] in counts:
            groups[(row["budget_scale"], row["gate"])].append(row)
    candidates = []
    for setting, rows in groups.items():
        seed_y = []
        for seed in SEEDS:
            selected = [r for r in rows if r["seed"] == seed]
            assert len(selected) == 4 * len(counts)
            seed_y.append(np.mean([r["weighted_success_rate"] for r in selected]))
        candidates.append(dict(setting=setting, x=np.mean([r["mean_cost"] for r in rows]),
                               y=np.mean(seed_y), seed_y=seed_y))
    ordered = sorted(candidates, key=lambda r: (r["x"], -r["y"]))
    collapsed = []
    for row in ordered:
        if collapsed and abs(row["x"] - collapsed[-1]["x"]) < 1e-9:
            if row["y"] > collapsed[-1]["y"]:
                collapsed[-1] = row
        else:
            collapsed.append(row)
    frontier = []
    best = -np.inf
    for row in collapsed:
        if row["y"] > best + 1e-9:
            frontier.append(row)
            best = row["y"]
    xs = np.array([r["x"] for r in frontier])
    low, high = max(0.4907343563068964, xs[0]), min(1.527859839700497, xs[-1])
    if low >= high:
        raise RuntimeError(f"Insufficient cost coverage: {policy}, counts={counts}")
    grid = np.unique(np.r_[np.linspace(low, high, 200), xs[(xs >= low) & (xs <= high)],
                           [1.0] if low <= 1.0 <= high else []])
    seed_curves = np.array([np.interp(grid, xs, [r["seed_y"][i] for r in frontier]) for i in range(3)])
    mean = seed_curves.mean(axis=0)
    ci = t.ppf(0.975, 2) * seed_curves.std(axis=0, ddof=1) / np.sqrt(3)
    at_one = float(mean[np.where(grid == 1.0)[0][0]]) if low <= 1 <= high else None
    return dict(x=grid, mean=mean, ci=ci, at_one=at_one, low=low, high=high)


def winners(curves):
    result = {}
    for family, methods in (("ML", ML_ORDER), ("RL", RL)):
        available = [p for p in POLICIES if p in methods and curves[p]["at_one"] is not None]
        assert available, f"No {family} policy covers cost 1"
        result[family] = max(available, key=lambda p: curves[p]["at_one"])
    return result


def draw(ax, curves, best, colors):
    highlighted = set(best.values())
    for policy in [p for p in POLICIES if p not in highlighted] + list(best.values()):
        c = curves[policy]
        selected = policy in highlighted
        ml = policy in ML_ORDER
        baseline = policy not in RL and not ml
        line_style = "--" if ml else ":" if baseline else "-"
        color = colors[policy]
        ax.plot(c["x"], c["mean"], color=color, linestyle=line_style,
                linewidth=3.4 if selected else 1.0, alpha=1.0 if selected else 0.44,
                zorder=7 if selected else 2)
        ax.fill_between(c["x"], np.clip(c["mean"]-c["ci"], 0, 1),
                        np.clip(c["mean"]+c["ci"], 0, 1), color=color,
                        alpha=0.12 if selected else 0.018, linewidth=0, zorder=1)
        if selected:
            ax.scatter([1.0], [c["at_one"]], s=34, color=color, edgecolor="white", linewidth=0.8, zorder=8)
    ax.axvline(1.0, color="#454545", linestyle="--", linewidth=1.0, zorder=3)
    ax.set_xlim(0.4907343563068964, 1.527859839700497)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_xlabel("Average normalized communication cost", labelpad=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#606060")


def limits(curve_sets):
    # Include displayed confidence bands with a small margin; no forced 0--1 axis.
    lows = [float(np.clip(c["mean"]-c["ci"], 0, 1).min()) for cs in curve_sets for c in cs.values()]
    highs = [float(np.clip(c["mean"]+c["ci"], 0, 1).max()) for cs in curve_sets for c in cs.values()]
    span = max(0.1, max(highs)-min(lows))
    return max(0, min(lows)-0.035*span), min(1, max(highs)+0.035*span)


def legend(fig, best_methods, colors, *, multiple):
    # Put winners first, bold their labels, and retain all remaining methods as context.
    highlighted = list(dict.fromkeys(best_methods))
    order = highlighted + [p for p in POLICIES if p not in highlighted]
    handles, labels = [], []
    for p in order:
        ml = p in ML_ORDER
        selected = p in highlighted
        handles.append(Line2D([], [], color=colors[p], linestyle="--" if ml else "-" if p in RL else ":",
                              linewidth=3.4 if selected else 1.0, alpha=1 if selected else 0.55))
        labels.append(LABELS.get(p, p))
    result = fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.995, 0.53),
                        fontsize=8.6 if multiple else 8.7, frameon=False,
                        handlelength=2.8, labelspacing=0.6, borderaxespad=0)
    for text, p in zip(result.get_texts(), order):
        if p in highlighted:
            text.set_fontweight("bold")


def save(stem, panels, data, colors):
    multiple = len(panels) > 1
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=(18.7, 5.3) if multiple else (11.3, 5.5))
    chosen = []
    for ax, (case, title) in zip(axes[0], panels):
        best = winners(data[case])
        chosen.extend(best.values())
        draw(ax, data[case], best, colors)
        ax.set_title(title, fontsize=11.5, pad=10)
        summary = "\n".join(f"Best {family} @1: {data[case][policy]['at_one']:.2%}" for family, policy in best.items())
        ax.text(0.025, 0.97, summary, transform=ax.transAxes, va="top", fontsize=8.9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=3))
    axes[0, 0].set_ylabel("Importance-weighted defense success rate", labelpad=8)
    axes[0, 0].set_ylim(*limits([data[case] for case, _ in panels]))
    legend(fig, chosen, colors, multiple=multiple)
    fig.text(0.5, 0.023,
             "Mean + pointwise 95% CI (3 evaluation seeds); bold = best ML/RL at cost 1.0; ML dashed, RL solid.",
             ha="center", fontsize=8.2, color="#555555")
    fig.subplots_adjust(left=0.055 if multiple else 0.075, right=0.835 if multiple else 0.755,
                        top=0.90, bottom=0.15, wspace=0.12)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    raw = read_rows()
    data = {case: {p: curve(raw, p, counts) for p in POLICIES} for case, counts in CASES.items()}
    # Keep the established colors for the two leading RL candidates.
    palette = plt.get_cmap("tab20")(np.linspace(0, 0.95, len(POLICIES)))
    colors = dict(zip(POLICIES, palette))
    colors["QR-DQN original"], colors["QR-DQN scratch multi"] = "#1f77b4", "#d62728"
    colors["Linear SVM"] = "#218739"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save("01_single_attacker_success_cost", [("single", "Single attacker: success-cost curves (mean and 95% CI)")], data, colors)
    save("02_multiple_attackers_success_cost", [("multi_2", "2 attackers"), ("multi_3", "3 attackers"), ("multi_5", "5 attackers")], data, colors)
    save("03_unknown_attackers_success_cost", [("unknown", "Unknown attacker count: equal mixture of 1 / 2 / 3 / 5")], data, colors)
    records = []
    for case, curves in data.items():
        best = winners(curves)
        for policy, c in curves.items():
            records.append(dict(case=case, policy=policy, weighted_success_at_cost_1=c["at_one"],
                                best_ml=policy == best["ML"], best_rl=policy == best["RL"],
                                min_plotted_cost=c["low"], max_plotted_cost=c["high"]))
        print(case, {family: (policy, round(100*curves[policy]["at_one"], 4)) for family, policy in best.items()})
    with (CACHE / "ml_rl_matched_cost_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (CACHE / "plot_protocol.json").write_text(json.dumps(dict(
        rank_cost=1.0, ci="Pointwise Student-t over 3 paired evaluation seeds, df=2; fixed selected working points and mean cost coordinates.",
        excludes="Training, frontier-selection and x-coordinate uncertainty; point estimates on evaluation data are descriptive rankings.",
        unknown_count_weights={str(c):0.25 for c in CASES["unknown"]},
        extrapolation=False, common_multiple_attacker_ylim=True,
        output_format="PNG only"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
