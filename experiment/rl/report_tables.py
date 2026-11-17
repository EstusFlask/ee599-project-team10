"""Turn the final analysis into markdown tables for the report (section 4 of final.md)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCENARIOS = ("independent", "waves", "campaign", "campaign_waves")
SHOW_COSTS = (4.0, 8.0, 12.0, 15.0, 20.0)


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def cell(row, key, ci=False) -> str:
    value = num(row.get(key))
    if math.isnan(value):
        return "—"
    text = f"{value:+.1f}"
    if ci:
        low, high = num(row.get(key.replace("_gap", "_ci_low"))), num(row.get(key.replace("_gap", "_ci_high")))
        if not (math.isnan(low) or math.isnan(high)):
            text += f" [{low:+.1f}, {high:+.1f}]"
    return text


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("experiment/results/rl_large_scale_final"))
    args = parser.parse_args()
    root = args.root
    analysis = list(csv.DictReader((root / "analysis.csv").open(encoding="utf-8")))
    by = {(r["scenario"], float(r["cost"])): r for r in analysis}
    lines = []

    lines.append("### 主结果：QR-DQN（3 个训练种子平均）、调参规则与冻结固定基线\n")
    lines.append("固定策略列为成功率；规则、RL 列为相对固定策略的差距（百分点，正数为更好），"
                 "RL 方括号为 95% 自助法区间（对 20 个测试种子重采样）。`—` 表示该方法在此成本下没有数据。\n")
    for scenario in SCENARIOS:
        rows = []
        for cost in SHOW_COSTS:
            r = by.get((scenario, cost))
            if r is None:
                continue
            fixed = num(r.get("fixed_success"))
            rows.append([f"{cost:g}", "—" if math.isnan(fixed) else f"{fixed:.3f}",
                         cell(r, "rule_gap"), cell(r, "rl_gap", ci=True),
                         "—" if not r.get("rl_prob_better") else f"{num(r['rl_prob_better']):.2f}",
                         "—" if not r.get("rl_seeds") else f"{r.get('rl_seeds_positive')}/{r.get('rl_seeds')}"])
        lines.append(f"**{scenario}**\n")
        lines.append(table(["平均成本", "固定策略成功率", "规则差距", "RL 差距 [95%区间]", "P(RL>固定)", "为正的种子数"], rows))
        lines.append("")

    other = sorted({k[:-4] for r in analysis for k in r
                    if k.endswith("_gap") and k not in ("rl_gap", "rule_gap")
                    and not k.startswith("rl_qrdqn_seed")})
    lines.append("### 其他 RL 算法、经典 ML 与 RL 消融（相对冻结固定策略的差距，百分点）\n")
    lines.append("每个模型 1 份（RL 消融与其他算法为训练种子 2026）；方括号为 95% 自助法区间。\n")
    for scenario in SCENARIOS:
        rows = []
        for name in ["rl_qrdqn_seed2026"] + other:
            cells = []
            for cost in SHOW_COSTS[:4]:
                r = by.get((scenario, cost))
                cells.append("—" if r is None else cell(r, f"{name}_gap", ci=name in other))
            if any(c != "—" for c in cells):
                rows.append([name] + cells)
        lines.append(f"**{scenario}**\n")
        lines.append(table(["模型"] + [f"成本 {c:g}" for c in SHOW_COSTS[:4]], rows))
        lines.append("")

    ablation = root / "rule_ablation" / "summary.csv"
    if ablation.exists():
        rows = []
        for r in csv.DictReader(ablation.open(encoding="utf-8")):
            rows.append([r["variant"]] + [f"{num(v):+.1f}" for k, v in r.items() if k != "variant"])
        header = ["规则变体"] + [k.replace("mean_gap_cost", "成本 ") for k in
                              next(csv.DictReader(ablation.open(encoding="utf-8"))) if k != "variant"]
        lines.append("### 规则消融（各场景平均，相对固定策略凸包的差距，百分点）\n")
        lines.append(table(header, rows))
        lines.append("")

    rule = root / "rule" / "chosen_receipt_delay1.json"
    if rule.exists():
        chosen = json.loads(rule.read_text())
        lines.append(f"### 调参规则：训练种子上入选的 {len(chosen)} 组参数\n")
        lines.append(table(["训练成本", "训练成功率"] + list(chosen[0]["params"]),
                           [[f"{c['train_cost']:.2f}", f"{c['train_success']:.3f}"]
                            + [f"{v:.3g}" for v in c["params"].values()] for c in chosen]))
        lines.append("")

    ml = root / "ml" / "models" / "training_report.json"
    if ml.exists():
        lines.append("### 经典 ML 模型（监督标签 = 已知真实被攻陷数时的最优冗余数）\n")
        lines.append(table(["模型", "验证集准确率", "训练秒数"],
                           [[r["model"], f"{r['validation_accuracy']:.3f}", f"{r['training_seconds']:.1f}"]
                            for r in json.loads(ml.read_text())]))
        lines.append("")

    runs = []
    for path in sorted(root.glob("**/*_seed*.json")):
        if path.name.endswith("budget.json"):
            continue
        meta = json.loads(path.read_text())
        runs.append([path.relative_to(root).with_suffix('').as_posix(), str(meta.get("algorithm")), str(meta.get("steps")),
                     f"{meta.get('training_seconds', 0) / 60:.1f}", str(meta.get("device_used"))])
    if runs:
        lines.append("### 训练记录\n")
        lines.append(table(["模型文件", "算法", "训练步数", "分钟", "设备"], runs))
        lines.append("")

    output = root / "report_tables.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
