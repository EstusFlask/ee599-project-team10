"""Plot a readable probe-lull-burst pattern from a real evaluation trace."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REGIME_NAMES = {0: "Quiet", 1: "Probing", 2: "Burst"}
REGIME_COLORS = {0: "#72b7b2", 1: "#f2cf5b", 2: "#e45756"}


def centered_mean(values, width=9):
    values = np.asarray(values, dtype=float)
    kernel = np.ones(width, dtype=float)
    return (
        np.convolve(values, kernel, mode="same")
        / np.convolve(np.ones_like(values), kernel, mode="same")
    )


def select_pattern(rows):
    trajectories = {}
    for row in rows:
        if row["policy"] != "Robust QR-DQN":
            continue
        trajectories.setdefault((row["scenario"], row["seed"]), []).append(row)

    candidates = []
    for (scenario, seed), trajectory in trajectories.items():
        trajectory.sort(key=lambda row: int(row["round"]))
        attacks = np.asarray([int(row["true_attacks"]) for row in trajectory])
        smooth = centered_mean(attacks)
        for lull_center in range(45, 151):
            probe_slice = slice(lull_center - 40, lull_center - 10)
            lull_slice = slice(lull_center - 7, lull_center + 8)
            probe_mean = float(smooth[probe_slice].mean())
            lull_mean = float(smooth[lull_slice].mean())
            for burst_center in range(lull_center + 25, min(lull_center + 91, 225)):
                burst_slice = slice(burst_center - 7, burst_center + 14)
                burst_mean = float(smooth[burst_slice].mean())
                score = (
                    1.1 * (probe_mean - lull_mean)
                    + 1.9 * (burst_mean - lull_mean)
                    + 0.35 * burst_mean
                    - 0.30 * abs(probe_mean - 1.45)
                )
                candidates.append((
                    score, scenario, seed, trajectory, lull_center, burst_center,
                    probe_mean, lull_mean, burst_mean,
                ))
    return max(candidates, key=lambda item: item[0])


def write_manifest(path, selected):
    (
        score, scenario, seed, _, lull_center, burst_center,
        probe_mean, lull_mean, burst_mean,
    ) = selected
    row = {
        "selection": "maximum probe-lull-burst contrast among existing robust QR-DQN traces",
        "scenario": scenario,
        "seed": seed,
        "lull_center": lull_center,
        "burst_center": burst_center,
        "probe_smoothed_mean": probe_mean,
        "lull_smoothed_mean": lull_mean,
        "burst_smoothed_mean": burst_mean,
        "selection_score": score,
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def make_figure(selected, output):
    (
        _, scenario, seed, trajectory, lull_center, burst_center,
        probe_mean, lull_mean, burst_mean,
    ) = selected
    rounds = np.asarray([int(row["round"]) for row in trajectory])
    attacks = np.asarray([int(row["true_attacks"]) for row in trajectory])
    alerts = np.asarray([int(row["alert_count"]) for row in trajectory])
    regimes = np.asarray([int(row["regime"]) for row in trajectory])
    smooth = centered_mean(attacks)

    start = max(0, lull_center - 48)
    probe_end = lull_center - 8
    onset_candidates = np.flatnonzero(
        smooth[lull_center + 5:burst_center + 1] > lull_mean + 1.0
    )
    burst_start = (
        lull_center + 5 + int(onset_candidates[0])
        if len(onset_candidates) else burst_center - 8
    )
    end = min(len(rounds) - 1, burst_center + 42)
    keep = (rounds >= start) & (rounds <= end)

    fig, (ax, regime_ax) = plt.subplots(
        2, 1, figsize=(14.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [7, 0.65], "hspace": 0.06},
    )
    ax.axvspan(start, probe_end, color="#f2cf5b", alpha=0.20)
    ax.axvspan(probe_end, burst_start, color="#72b7b2", alpha=0.20)
    ax.axvspan(burst_start, end, color="#e45756", alpha=0.16)
    ax.bar(rounds[keep], attacks[keep], width=0.86, color="#4c78a8", alpha=0.42,
           label="Compromised replicas (raw)")
    ax.plot(rounds[keep], smooth[keep], color="#d62728", linewidth=3.0,
            label="Attack intensity (9-round mean)")
    ax.plot(rounds[keep], alerts[keep], color="#f58518", linewidth=1.4,
            linestyle="--", alpha=0.85, label="Noisy detector alert")

    phase_y = 4.72
    ax.annotate("1  Reconnaissance / probing", xy=((start + probe_end) / 2, phase_y),
                ha="center", va="center", fontsize=11, weight="bold")
    ax.annotate("2  Temporary lull", xy=((probe_end + burst_start) / 2, phase_y),
                ha="center", va="center", fontsize=11, weight="bold")
    ax.annotate("3  Sudden burst", xy=((burst_start + end) / 2, phase_y),
                ha="center", va="center", fontsize=11, weight="bold")
    ax.annotate(
        f"probe mean {probe_mean:.2f}",
        xy=(probe_end - 9, smooth[probe_end - 9]), xytext=(probe_end - 31, 3.65),
        arrowprops={"arrowstyle": "->", "color": "#6b5d00"}, fontsize=10,
    )
    ax.annotate(
        f"lull {lull_mean:.2f}",
        xy=(lull_center, smooth[lull_center]), xytext=(lull_center - 4, 0.35),
        arrowprops={"arrowstyle": "->", "color": "#246b68"}, fontsize=10,
    )
    spike_index = burst_start + int(np.argmax(smooth[burst_start:burst_center + 14]))
    ax.annotate(
        f"burst mean {burst_mean:.2f}",
        xy=(spike_index, smooth[spike_index]), xytext=(spike_index + 8, 3.85),
        arrowprops={"arrowstyle": "->", "color": "#9d2424"}, fontsize=10,
    )
    ax.axvline(probe_end, color="#888888", linestyle=":", linewidth=1.2)
    ax.axvline(burst_start, color="#888888", linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 5.18)
    ax.set_ylabel("Compromised replicas / alert count")
    ax.grid(True, axis="y", alpha=0.20)
    handles, labels = ax.get_legend_handles_labels()

    for round_index in rounds[keep]:
        regime_ax.axvspan(
            round_index - 0.5, round_index + 0.5,
            color=REGIME_COLORS[regimes[round_index]], linewidth=0,
        )
    regime_ax.set_yticks([])
    regime_ax.set_ylabel("Hidden\nregime", rotation=0, labelpad=30, va="center")
    regime_ax.set_xlabel("Round")
    regime_ax.set_xlim(start - 0.5, end + 0.5)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[index], label=REGIME_NAMES[index])
        for index in (0, 1, 2)
    ]
    regime_ax.legend(handles=legend_handles, loc="lower right", ncol=3,
                     bbox_to_anchor=(1.0, -0.08), fontsize=8.5)
    for spine in regime_ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Representative Multi-stage Attack: Probe -> Lull -> Burst",
        y=0.982, fontsize=17, weight="bold",
    )
    fig.text(
        0.5, 0.925,
        f"Actual held-out trajectory: {scenario} scenario, seed {seed}, robust QR-DQN defense",
        ha="center", fontsize=10.5,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               ncol=3, fontsize=9.2, framealpha=0.95)
    fig.text(
        0.5, 0.012,
        "Representative real trace, selected automatically for the strongest probe-lull-burst "
        "contrast among the 15 benchmark trajectories; this is a possible pattern, not a fixed schedule.",
        ha="center", fontsize=9.2,
    )
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.07, right=0.985, hspace=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("experiment/results/rl_robust_w015/analysis/attacker_dynamics_raw.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/results/rl_robust_w015/analysis/attack_pattern_example.png"),
    )
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_pattern(rows)
    make_figure(selected, args.output)
    write_manifest(args.output.with_name("attack_pattern_example.csv"), selected)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
