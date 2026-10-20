"""Train and compare one baseline and three defense models.

All policies receive the same average repeat budget. Resource and failure costs
are configured once at the top of this file. No LLM or text is required.
"""

# All cost/reward settings are defined only here.
cost = {
    # The defense sends full text on at most two carriers; remaining packets are
    # SHA-256 hex witnesses (64 bytes). Reference: core/engine.py.
    "text_carriers": 2,
    "hash_bytes": 64,
    # Reference extra bytes: k=3 at the reference message length. Its
    # normalized communication cost is exactly 1.
    "reference_message_length": 100,
    "reference_k": 3,
    # One fresh transmission is attempted after an initial quorum failure.
    "max_retries": 1,
    # https://arxiv.org/abs/2304.11084
    # Nyberg & Johnson (2023) separate defense cost c_d and failure cost c_f.
    # Base case normalizes a normal failure and reference defense to the same 1.0.
    "fail": 1.0,
    # https://www.first.org/cvss/v3-1/specification-document
    # CVSS v3.1 Security Requirement constants: Low=.5, Medium=1, High=1.5.
    "importance": {"low": 0.5, "normal": 1.0, "high": 1.5},
    "message_lengths": (50, 100, 200),
    "message_length_probability": (0.30, 0.50, 0.20),
    # Persistent workload states and gradual length changes simulate a running
    # agent whose context expands and contracts over consecutive turns.
    "message_length_transition": (
        (0.75, 0.20, 0.05),
        (0.15, 0.70, 0.15),
        (0.05, 0.20, 0.75),
    ),
    "message_length_smoothing": 0.65,
    "message_length_jitter": 0.25,
    "importance_probability": {"low": 0.30, "normal": 0.50, "high": 0.20},
}

reward = {
    "mode": "equal_cost",  # budget, equal_cost, or utility
    "success": 1.0,
    "defense_weight": 1.0,
}

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

try:
    from .attacker import attacker, multiattacker
    from .defender import (
        ACTIONS,
        MLdefender_nostate,
        MLdefender_withstate,
        RLdefender,
        defense_cost,
        defense_success,
        fixdefender,
    )
except ImportError:
    from attacker import attacker, multiattacker
    from defender import (
        ACTIONS,
        MLdefender_nostate,
        MLdefender_withstate,
        RLdefender,
        defense_cost,
        defense_success,
        fixdefender,
    )


MAX_COST = max(defense_cost(action) for action in ACTIONS)


def transmission_bytes(action, message_length):
    text_packets = min(int(action), cost["text_carriers"])
    witness_packets = max(0, int(action) - text_packets)
    return (
        text_packets * float(message_length)
        + witness_packets * cost["hash_bytes"]
    )


def calculate_cost(
    action,
    success,
    message_length,
    importance,
    retries=0,
):
    """Normalize actual extra/retry bytes and add final failure impact."""
    baseline_bytes = transmission_bytes(1, message_length)
    protected_bytes = transmission_bytes(action, message_length)
    initial_extra_bytes = max(0.0, protected_bytes - baseline_bytes)
    retry_bytes = max(0, int(retries)) * protected_bytes
    reference_extra_bytes = (
        transmission_bytes(cost["reference_k"], cost["reference_message_length"])
        - transmission_bytes(1, cost["reference_message_length"])
    )
    communication = (initial_extra_bytes + retry_bytes) / reference_extra_bytes
    costs = {
        "initial_extra_bytes": initial_extra_bytes,
        "retry_bytes": retry_bytes,
        "communication": communication,
        "failure": (
            0.0
            if success
            else cost["fail"] * cost["importance"][importance]
        ),
    }
    costs["defense"] = costs["communication"]
    costs["total"] = costs["defense"] + costs["failure"]
    return costs


def calculate_reward(
    success,
    costs,
):
    """Reward alternatives kept together for controlled experiments.

    budget:
        Maximize success under the enforced average defense-cost budget.
    equal_cost:
        Defense and error both have unit weight, as requested for the base case.
    utility:
        Positive success reward minus weighted defense and error costs.
    """
    mode = reward["mode"]
    error = costs["failure"]
    defense = reward["defense_weight"] * costs["defense"]
    if mode == "budget":
        return reward["success"] if success else -error
    if mode == "equal_cost":
        return -defense - error
    if mode == "utility":
        return (reward["success"] if success else 0.0) - defense - error
    raise ValueError(f"unknown reward mode: {mode}")


def minimum_cost_defense(attack_count, message_length, importance):
    """Offline supervised label using the known simulated attack outcome."""
    return min(
        ACTIONS,
        key=lambda action: (
            calculate_cost(
                action=action,
                success=defense_success(attack_count, action),
                message_length=message_length,
                importance=importance,
            )["total"],
            not defense_success(attack_count, action),
            action,
        ),
    )


def make_attacker(seed, method, profile, attackers, steps):
    """Create one attacker or staggered independent attackers.

    Later attackers start near evenly spaced points with per-episode jitter, so
    test episodes contain unseen start times instead of one memorized schedule.
    """
    if attackers <= 1:
        return attacker(
            seed=seed,
            method=method,
            profile=profile,
            max_attack=3,
        )
    start_generator = random.Random(f"{seed}:start-times")
    jitter = max(1, steps // 12)
    start_times = [0]
    for index in range(1, attackers):
        center = round(index * steps / attackers)
        start_times.append(
            max(1, min(steps - 1, center + start_generator.randint(-jitter, jitter)))
        )
    return multiattacker(
        start_times=tuple(start_times),
        seed=seed,
        method=method,
        profile=profile,
        max_attack=3,
    )


def allowed_actions(remaining_budget, rounds_left):
    """Actions that still allow the episode to finish at exactly its budget."""
    if rounds_left <= 0:
        return ACTIONS
    max_future_cost = MAX_COST * (rounds_left - 1)
    minimum_now = max(0, remaining_budget - max_future_cost)
    maximum_now = min(MAX_COST, remaining_budget)
    choices = tuple(
        action
        for action in ACTIONS
        if minimum_now <= defense_cost(action) <= maximum_now
    )
    if not choices:
        raise RuntimeError("no action can satisfy the remaining budget")
    return choices


def sample_task_context(generator):
    message_length = generator.choices(
        cost["message_lengths"],
        weights=cost["message_length_probability"],
        k=1,
    )[0]
    importance_names = tuple(cost["importance_probability"])
    importance = generator.choices(
        importance_names,
        weights=tuple(cost["importance_probability"].values()),
        k=1,
    )[0]
    return message_length, importance


def make_task_contexts(generator, steps):
    """Temporally correlated message lengths for one continuous agent run."""
    length_state = 1
    current_length = float(cost["reference_message_length"])
    importance_names = tuple(cost["importance_probability"])
    contexts = []
    for _ in range(steps):
        length_state = generator.choices(
            range(len(cost["message_lengths"])),
            weights=cost["message_length_transition"][length_state],
            k=1,
        )[0]
        base_length = cost["message_lengths"][length_state]
        jitter = cost["message_length_jitter"]
        target_length = base_length * generator.uniform(1.0 - jitter, 1.0 + jitter)
        smoothing = cost["message_length_smoothing"]
        current_length = smoothing * current_length + (1.0 - smoothing) * target_length
        message_length = max(20, int(round(current_length)))
        importance = generator.choices(
            importance_names,
            weights=tuple(cost["importance_probability"].values()),
            k=1,
        )[0]
        contexts.append((message_length, importance))
    return contexts


def observation(
    history,
    remaining_budget,
    rounds_left,
    message_length=100,
    importance="normal",
    total_rounds=None,
):
    total_rounds = max(1, int(total_rounds or rounds_left))
    return {
        "history": list(history),
        "remaining_budget": remaining_budget / max(1, rounds_left),
        "allowed_actions": allowed_actions(remaining_budget, rounds_left),
        "progress": 1.0 - rounds_left / total_rounds,
        "message_length": message_length,
        "importance": importance,
    }


def make_supervised_data(
    episodes,
    steps,
    method,
    profile,
    window,
    seed,
    budget,
    attackers=1,
):
    observations = []
    labels = []
    episode_budget = int(round(budget * steps))
    for episode in range(episodes):
        source = make_attacker(
            seed=seed + episode,
            method=method,
            profile=profile,
            attackers=attackers,
            steps=steps,
        )
        task_generator = random.Random(seed + 30_000 + episode)
        task_contexts = make_task_contexts(task_generator, steps)
        behavior_generator = random.Random(seed + 60_000 + episode)
        history = deque(maxlen=window)
        recent_success = deque([1.0], maxlen=window)
        remaining_budget = episode_budget
        for step in range(steps):
            rounds_left = steps - step
            message_length, importance = task_contexts[step]
            current = observation(
                history,
                remaining_budget,
                rounds_left,
                message_length,
                importance,
                steps,
            )
            success_rate = sum(recent_success) / len(recent_success)
            event = source.nextstate(success_rate=success_rate)
            observations.append(current)
            target_action = minimum_cost_defense(
                event["attack_count"],
                message_length,
                importance,
            )
            labels.append(
                min(
                    current["allowed_actions"],
                    key=lambda action: (abs(action - target_action), action),
                )
            )
            history.append(event["attack_count"])
            behavior_action = behavior_generator.choice(current["allowed_actions"])
            remaining_budget -= defense_cost(behavior_action)
            recent_success.append(
                float(defense_success(event["attack_count"], behavior_action))
            )
    return observations, labels


def train_rl(
    model,
    episodes,
    steps,
    budget,
    method,
    profile,
    window,
    seed,
    attackers=1,
):
    episode_budget = int(round(budget * steps))
    if not 0 <= episode_budget <= MAX_COST * steps:
        raise ValueError(f"budget must be between 0 and {MAX_COST}")

    for episode in range(episodes):
        source = make_attacker(
            seed=seed + 10_000 + episode,
            method=method,
            profile=profile,
            attackers=attackers,
            steps=steps,
        )
        task_generator = random.Random(seed + 40_000 + episode)
        task_contexts = make_task_contexts(task_generator, steps)
        model.reset()
        model.epsilon = max(0.02, 0.25 * (1.0 - episode / max(1, episodes)))
        history = deque(maxlen=window)
        recent_success = deque([1.0], maxlen=window)
        remaining_budget = episode_budget

        for step in range(steps):
            rounds_left = steps - step
            message_length, importance = task_contexts[step]
            current = observation(
                history,
                remaining_budget,
                rounds_left,
                message_length,
                importance,
                steps,
            )
            action = model.predict(current, explore=True)
            remaining_budget -= defense_cost(action)

            success_rate = sum(recent_success) / len(recent_success)
            event = source.nextstate(success_rate=success_rate)
            success = defense_success(event["attack_count"], action)
            observed_attack = event["attack_count"]
            retries = 0
            while not success and retries < cost["max_retries"]:
                retry_event = source.retry(success_rate=success_rate)
                retries += 1
                observed_attack = max(
                    observed_attack,
                    retry_event["attack_count"],
                )
                success = defense_success(retry_event["attack_count"], action)
            costs = calculate_cost(
                action=action,
                success=success,
                message_length=message_length,
                importance=importance,
                retries=retries,
            )
            step_reward = calculate_reward(success=success, costs=costs)

            history.append(observed_attack)
            model.observe(observed_attack)
            recent_success.append(float(success))
            done = step == steps - 1
            if done:
                next_state = {"history": list(history)}
            else:
                next_length, next_importance = task_contexts[step + 1]
                next_state = observation(
                    history,
                    remaining_budget,
                    rounds_left - 1,
                    next_length,
                    next_importance,
                    steps,
                )
            model.update(next_state, step_reward, done=done)
    model.epsilon = 0.0
    return model


def evaluate_policy(
    name,
    model,
    episodes,
    steps,
    budget,
    method,
    profile,
    window,
    seed,
    clean_accuracy,
    attackers=1,
):
    episode_budget = int(round(budget * steps))
    total_rounds = episodes * steps
    failures = 0
    attack_rounds = 0
    attack_failures = 0
    total_budget_units = 0
    total_defense_cost = 0.0
    total_failure_cost = 0.0
    total_message_length = 0
    minimum_message_length = None
    maximum_message_length = None
    total_length_change = 0
    length_change_count = 0

    for episode in range(episodes):
        source = make_attacker(
            seed=seed + 20_000 + episode,
            method=method,
            profile=profile,
            attackers=attackers,
            steps=steps,
        )
        task_generator = random.Random(seed + 50_000 + episode)
        task_contexts = make_task_contexts(task_generator, steps)
        if hasattr(model, "reset"):
            model.reset()
        history = deque(maxlen=window)
        recent_success = deque([1.0], maxlen=window)
        remaining_budget = episode_budget
        previous_length = None

        for step in range(steps):
            rounds_left = steps - step
            message_length, importance = task_contexts[step]
            total_message_length += message_length
            minimum_message_length = (
                message_length
                if minimum_message_length is None
                else min(minimum_message_length, message_length)
            )
            maximum_message_length = (
                message_length
                if maximum_message_length is None
                else max(maximum_message_length, message_length)
            )
            if previous_length is not None:
                total_length_change += abs(message_length - previous_length)
                length_change_count += 1
            previous_length = message_length
            current = observation(
                history,
                remaining_budget,
                rounds_left,
                message_length,
                importance,
                steps,
            )
            action = model.predict(current)
            budget_units = defense_cost(action)
            remaining_budget -= budget_units
            total_budget_units += budget_units

            success_rate = sum(recent_success) / len(recent_success)
            event = source.nextstate(success_rate=success_rate)
            attack_count = event["attack_count"]
            success = defense_success(attack_count, action)
            observed_attack = attack_count
            retries = 0
            while not success and retries < cost["max_retries"]:
                retry_event = source.retry(success_rate=success_rate)
                retries += 1
                observed_attack = max(
                    observed_attack,
                    retry_event["attack_count"],
                )
                success = defense_success(retry_event["attack_count"], action)
            step_costs = calculate_cost(
                action=action,
                success=success,
                message_length=message_length,
                importance=importance,
                retries=retries,
            )
            total_defense_cost += step_costs["defense"]
            total_failure_cost += step_costs["failure"]
            failures += int(not success)
            if attack_count > 0:
                attack_rounds += 1
                attack_failures += int(not success)

            history.append(observed_attack)
            if hasattr(model, "observe"):
                model.observe(observed_attack)
            recent_success.append(float(success))

        if remaining_budget != 0:
            raise RuntimeError("policy did not use the exact episode budget")

    integrity_success = 1.0 - failures / total_rounds
    return {
        "model": name,
        "average_repeat_budget": total_budget_units / total_rounds,
        "average_communication_cost": total_defense_cost / total_rounds,
        "average_defense_cost": total_defense_cost / total_rounds,
        "average_failure_cost": total_failure_cost / total_rounds,
        "average_total_cost": (
            total_defense_cost + total_failure_cost
        ) / total_rounds,
        "integrity_success_rate": integrity_success,
        "attack_success_rate": (
            attack_failures / attack_rounds if attack_rounds else 0.0
        ),
        "estimated_downstream_accuracy": clean_accuracy * integrity_success,
        "rounds": total_rounds,
        "attacked_rounds": attack_rounds,
        "average_message_length": total_message_length / total_rounds,
        "minimum_message_length": minimum_message_length,
        "maximum_message_length": maximum_message_length,
        "average_absolute_length_change": (
            total_length_change / length_change_count if length_change_count else 0.0
        ),
    }


def print_results(results):
    headers = (
        "model",
        "repeat",
        "comm",
        "failure",
        "total",
        "ASR",
        "estimated_acc",
    )
    print(
        f"{headers[0]:<22} {headers[1]:>7} {headers[2]:>8} "
        f"{headers[3]:>8} {headers[4]:>8} {headers[5]:>9} {headers[6]:>14}"
    )
    print("-" * 91)
    for result in results:
        print(
            f"{result['model']:<22} "
            f"{result['average_repeat_budget']:>7.3f} "
            f"{result['average_communication_cost']:>8.3f} "
            f"{result['average_failure_cost']:>8.3f} "
            f"{result['average_total_cost']:>8.3f} "
            f"{result['attack_success_rate']:>9.3%} "
            f"{result['estimated_downstream_accuracy']:>14.3%}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacker", choices=("hmm", "ibal"), default="hmm")
    parser.add_argument(
        "--profile",
        choices=("stealth", "bursty", "persistent"),
        default="bursty",
    )
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--attackers", type=int, default=1)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--clean-accuracy", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_experiment(args, display=True):
    training_start = time.perf_counter()
    supervised_x, supervised_y = make_supervised_data(
        episodes=args.train_episodes,
        steps=args.steps,
        method=args.attacker,
        profile=args.profile,
        window=args.window,
        seed=args.seed,
        budget=args.budget,
        attackers=args.attackers,
    )

    no_state = MLdefender_nostate().train(supervised_x, supervised_y)
    with_state = MLdefender_withstate(window=args.window).train(
        supervised_x,
        supervised_y,
    )
    rl_model = train_rl(
        RLdefender(window=args.window, seed=args.seed),
        episodes=args.train_episodes,
        steps=args.steps,
        budget=args.budget,
        method=args.attacker,
        profile=args.profile,
        window=args.window,
        seed=args.seed,
        attackers=args.attackers,
    )
    training_seconds = time.perf_counter() - training_start

    baseline_action = min(
        ACTIONS,
        key=lambda action: abs(defense_cost(action) - args.budget),
    )
    models = (
        ("fixed_baseline", fixdefender(baseline_action)),
        ("ml_no_state", no_state),
        ("ml_with_state", with_state),
        ("q_learning_rl", rl_model),
    )
    testing_start = time.perf_counter()
    results = [
        evaluate_policy(
            name=name,
            model=model,
            episodes=args.test_episodes,
            steps=args.steps,
            budget=args.budget,
            method=args.attacker,
            profile=args.profile,
            window=args.window,
            seed=args.seed,
            clean_accuracy=args.clean_accuracy,
            attackers=args.attackers,
        )
        for name, model in models
    ]
    testing_seconds = time.perf_counter() - testing_start
    timing = {
        "training_seconds": training_seconds,
        "testing_seconds": testing_seconds,
        "total_seconds": training_seconds + testing_seconds,
    }
    if display:
        print_results(results)
        print(
            f"training={training_seconds:.3f}s "
            f"testing={testing_seconds:.3f}s "
            f"total={timing['total_seconds']:.3f}s"
        )

    payload = {
        "config": vars(args)
        | {
            "output": str(args.output) if args.output else None,
            "cost": cost,
            "reward": reward,
        },
        "timing": timing,
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    return payload


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
