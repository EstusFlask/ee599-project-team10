# Adaptive redundancy on the real defense engine

New code only; no existing file is modified. The real `core.engine.DefenseEngine` does the hashing and
the majority vote, a mock adversary replaces the LLM, and a learned or rule-based controller chooses the
number of channels `k` for every message.

| File | Role |
|---|---|
| `adaptive_engine.py` | `DefenseEngine` subclass: `set_k`, byte accounting, `integrity_verified()` |
| `mock_adversary.py` | LLM-free adversary with the same `manipulate` interface |
| `controller.py` | `FixedController`, `RuleController`, `QRDQNController`; use only deployable signals |
| `adapter.py` | `BaseDefenseAdapter` subclass; picks `k` before `secure_transmit` and feeds back the vote and receipt |
| `run_demo.py` | Runs policies through the engine under the simulated attack process |
| `tests/test_integration.py` | Engine vs. majority rule, byte cost, and controller observation == training environment |

**Success** means the hash that wins the vote is the hash of the message the sender transmitted, i.e. fewer
than half of the channels were altered. The receiver can then check any received text against that hash and
knows whether it was modified. It does not require the original text to arrive (only 2 channels carry full
text, so no budget could guarantee that). If a poisoned hash wins the vote, the round fails.

Signals a controller uses: message length, message importance, the size of the minority hash group
(`len(traitors)`) of past rounds, and a signed receipt (the sender compares the winning hash with the hash of
what it sent) that arrives one round late and is not charged. It never reads the attacker count or attack stage.

`max_ghost_channels = 0`: ghost rotation is not used by the policies.

```bash
python -m pytest integration/tests -q
python -m integration.run_demo --rule-params experiment/results/rl_large_scale_final/rule/chosen_receipt_delay1.json \
    --model experiment/results/rl_large_scale_final/models/qrdqn_seed2026.zip
```

To attach to a framework, construct `AdaptiveAdapter` instead of the base adapter and call `apply(agent)` from
the existing AutoGen / CAMEL / MetaGPT adapter; they all route through `secure_transmit`. In a real
deployment replace `attack_model` with the actual channel conditions (it only exists to drive the simulation).
