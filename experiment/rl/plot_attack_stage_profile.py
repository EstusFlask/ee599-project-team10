"""Plot the model's canonical probe-lull-burst stage profile.

This is a mechanism figure conditioned on the three hidden regimes.  It uses
the exact base compromise probabilities from ``DefenseEnv`` and imposes
an explanatory stage order; the stochastic benchmark does not force this
order in every episode.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


MAX_REPLICAS = 5
STAGES = (
    # name, start, end, per-replica compromise probability, color
    ("Reconnaissance / probing", 0, 60, 0.22, "#f2cf5b"),
    ("Temporary lull", 60, 82, 0.025, "#72b7b2"),
    ("Coordinated burst", 82, 142, 0.62, "#e45756"),
)


def stage_rows():
    rows = []
    for stage, start, end, probability, _ in STAGES:
        expected = MAX_REPLICAS * probability
        event_probability = 1.0 - (1.0 - probability) ** MAX_REPLICAS
        for round_index in range(start, end):
            rows.append({
                "round": round_index,
                "stage": stage,
                "per_replica_compromise_probability": probability,
                "expected_compromised_replicas": expected,
                "probability_any_compromise": event_probability,
            })
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows, output: Path):
    rounds = np.asarray([row["round"] for row in rows])
    expected = np.asarray([row["expected_compromised_replicas"] for row in rows])
    event_probability = np.asarray([row["probability_any_compromise"] for row in rows])

    fig, (ax_count, ax_event) = plt.subplots(
        2, 1, figsize=(14.5, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.6], "hspace": 0.08},
    )
    for stage_index, (stage, start, end, probability, color) in enumerate(STAGES, 1):
        for axis in (ax_count, ax_event):
            axis.axvspan(start, end, color=color, alpha=0.22, linewidth=0)
        expected_count = MAX_REPLICAS * probability
        any_event = 1.0 - (1.0 - probability) ** MAX_REPLICAS
        ax_count.text(
            (start + end) / 2, 4.45,
            f"{stage_index}  {stage}\nper-replica p = {probability:.3f}",
            ha="center", va="center", fontsize=11.5, weight="bold",
        )
        ax_count.annotate(
            f"Expected count = {expected_count:.3f}",
            xy=((start + end) / 2, expected_count),
            xytext=((start + end) / 2, expected_count + (0.62 if stage_index != 2 else 0.55)),
            ha="center", fontsize=10.5,
            arrowprops={"arrowstyle": "->", "color": "#444444"},
        )
        ax_event.text(
            (start + end) / 2, any_event + (0.07 if stage_index != 3 else -0.13),
            f"{any_event:.1%}", ha="center", fontsize=11, weight="bold",
        )

    ax_count.step(rounds, expected, where="post", color="#b22222", linewidth=4,
                  label="Expected compromised replicas")
    ax_count.fill_between(rounds, 0, expected, step="post", color="#b22222", alpha=0.10)
    ax_count.set_ylim(0, 5)
    ax_count.set_ylabel("Expected compromised replicas")
    ax_count.grid(True, axis="y", alpha=0.20)
    ax_count.legend(loc="upper left", fontsize=10)

    ax_event.step(rounds, event_probability, where="post", color="#4c78a8", linewidth=3.2,
                  label="Probability of >=1 compromised replica")
    ax_event.fill_between(rounds, 0, event_probability, step="post", color="#4c78a8", alpha=0.10)
    ax_event.set_ylim(0, 1.08)
    ax_event.set_ylabel("Attack-event probability")
    ax_event.set_xlabel("Round in the illustrative attack cycle")
    ax_event.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_event.grid(True, axis="y", alpha=0.20)
    ax_event.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "Canonical Multi-stage Attacker Profile: Probe -> Lull -> Burst",
        y=0.97, fontsize=17, weight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Mechanism figure using the environment's exact quiet/probing/burst base probabilities. "
        "The phase order is illustrative; the benchmark HMM samples its order stochastically.",
        ha="center", fontsize=9.5,
    )
    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.085, right=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/results/rl_robust_w015/analysis/attack_stage_profile.png"),
    )
    args = parser.parse_args()
    rows = stage_rows()
    make_figure(rows, args.output)
    write_csv(args.output.with_name("attack_stage_profile.csv"), rows)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
