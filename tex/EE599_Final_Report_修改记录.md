# EE599 期末报告修改记录

2026-11-23。本轮以 `tex/EE599_Final_Report/main.tex` 为底稿，更新了论文正文、图表说明和 Overleaf 提交包。第一版保存在 `tex/EE599_Final_Report_v1`。本轮沿用现有实验结果。

| 修改位置（main.tex 行号） | 本轮修改 | 核对依据 |
|---|---|---|
| Figure 1 图注与反馈定义（65、111） | 将回执描述为仿真中假定可信的完整性反馈。第 t 轮决策时，上一轮的投票异议已可用，最近的回执对应第 t−2 轮。 | `integration/adapter.py` 的 `secure_transmit` 调用 `integrity_verified()`；`experiment/rl/large_scale.py` 按反馈延迟索引成功记录。 |
| 固定策略前沿（135） | 写清在验证种子上选凸包顶点并冻结候选集合，在测试种子上重新测量这些候选，在测试前沿的相邻顶点间插值。相邻顶点可以跨过部分奇数 k。 | `experiment/rl/analyze_large_scale.py` 中的 `hull_vertices`、`hull_curve` 和 `gaps`。 |
| 对比算法设置（137） | 区分 DQN、PPO、A2C 的两层 128 单元网络与 RecurrentPPO 的单层 128 单元 LSTM、各含 96 个单元的策略头和价值头。120k 回放池只用于对比算法中的 DQN。 | `experiment/rl/train_large_scale.py` 的训练入口及 `train_agents.py` 的 `build_model`。 |
| 真实 LLM 静态实验（148） | 使用可追溯的双智能体 Coder/Reviewer 工作流描述，保留模型、427 个 MBPP 任务、攻击比例及静态防御结果。当前材料未能确认这次历史运行选用的适配器。 | `document/EE599_Milestone.pdf` 第 2 页明确记载实验条件；`benchmark.py` 的默认适配器只能说明默认配置。 |
| Independent 结果与结论（173、244） | 将“泛化更好”限定为留出测试种子上的具体比较。在成本 4 和 8，QR-DQN 相对固定基线分别为 +10.6 和 +9.7 pp，调参规则分别为 −6.1 和 −2.7 pp。成本 15 时 QR-DQN 比固定基线低 1.2 pp。 | `document/final.md` §5.2–5.3 的训练分布、种子划分和实验结果。 |
| 单种子比较（197） | 将 +11.8 pp 与三种子均值 +9.7 pp 之间的 2.1 pp 写为单次结果与均值的差。该差值本身不能估计种子间的离散程度。 | `document/final.md` §5.3–5.4。 |
| Table 4 正文、表注及结论（211、215、244） | 明确每格只平均该变体在该成本有覆盖的场景。各变体以相同随机种子分别搜索 150 个候选。列出低预算倒挂、成本 12 的回执差异及零延迟 −13.5 pp，删去未经重复搜索测量的 ±1–2 pp 噪声幅度。有限搜索和成本覆盖均作为可能因素。结论指明回执消融对应的算法和成本。 | `experiment/rl/tune_rule.py` 使用固定搜索种子 7，并跳过无覆盖的成本；`run_rule_ablations.py` 的 `summarize` 按可用行求均值。数值见 `document/final.md` §5.6–5.7。 |

## 文件与验证

- 更新 `tex/EE599_Final_Report/main.tex` 并重新生成 `tex/EE599_Final_Report.zip`。已检查包内 13 个文件可解压，内容与报告目录逐文件一致。
- 正文引用的 22 个文献键均有对应条目，交叉引用目标均存在。
- 本轮未重跑训练或 bootstrap。当前项目目录缺少逐种子原始结果，无法补算 Table 4 每格覆盖的场景数。
- 当前环境未找到 LaTeX 编译工具，最终 PDF 的分页和课程六页正文上限尚未验证。
