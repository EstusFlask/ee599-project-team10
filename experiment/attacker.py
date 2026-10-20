"""Numerical attackers used by the adaptive-defense experiment.

No text or LLM is needed here. Each call returns the number of logical
channels compromised in the current round.

``hmm`` follows the hidden-stage idea used by multi-stage attack literature:
https://doi.org/10.1155/2020/1428056

``ibal`` reproduces the success-conditioned attack-probability curriculum in
IBAL, Section 4.2: https://arxiv.org/abs/2605.18024
"""

import math
import random


class attacker:
    STATES = ("quiet", "probing", "burst")

    _PROFILES = {
        "stealth": {
            "transition": (
                (0.94, 0.055, 0.005),
                (0.20, 0.75, 0.05),
                (0.25, 0.55, 0.20),
            ),
            "attack_probability": (0.01, 0.15, 0.40),
            "attack_lambda": (0.0, 0.4, 1.2),
        },
        "bursty": {
            "transition": (
                (0.90, 0.09, 0.01),
                (0.10, 0.75, 0.15),
                (0.05, 0.10, 0.85),
            ),
            "attack_probability": (0.02, 0.20, 0.75),
            "attack_lambda": (0.0, 0.5, 2.0),
        },
        "persistent": {
            "transition": (
                (0.75, 0.20, 0.05),
                (0.08, 0.72, 0.20),
                (0.05, 0.15, 0.80),
            ),
            "attack_probability": (0.05, 0.35, 0.75),
            "attack_lambda": (0.1, 0.8, 1.8),
        },
    }

    def __init__(
        self,
        seed=None,
        method="hmm",
        profile="bursty",
        max_attack=3,
        ibal_alpha=1.1,
        ibal_threshold=0.8,
        adaptive_pressure=False,
    ):
        if method not in {"hmm", "ibal"}:
            raise ValueError("method must be 'hmm' or 'ibal'")
        if profile not in self._PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        if max_attack < 1:
            raise ValueError("max_attack must be at least 1")

        self.time = 0
        self.seed = random.random() if seed is None else seed
        self.random = random.Random(self.seed)
        self.retry_random = random.Random(f"{self.seed}:retry")
        self.method = method
        self.profile = profile
        self.max_attack = int(max_attack)
        self.ibal_alpha = float(ibal_alpha)
        self.ibal_threshold = float(ibal_threshold)
        self.adaptive_pressure = bool(adaptive_pressure)
        self.state = "quiet"
        self.p_max = 1.0 / self.max_attack

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.random = random.Random(self.seed)
        self.retry_random = random.Random(f"{self.seed}:retry")
        self.time = 0
        self.state = "quiet"
        self.p_max = 1.0 / self.max_attack

    def _sample_index(self, probabilities):
        draw = self.random.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return index
        return len(probabilities) - 1

    def _poisson(self, rate):
        if rate <= 0:
            return 0
        threshold = math.exp(-rate)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= self.random.random()
        return count - 1

    def _emit_hmm(self, advance_state, success_rate=None):
        config = self._PROFILES[self.profile]
        if advance_state:
            current_index = self.STATES.index(self.state)
            transition = list(config["transition"][current_index])
            if self.adaptive_pressure and success_rate is not None:
                # Fixed heuristic, not a learning attacker: low recent defense
                # success shifts transition mass toward probing and burst.
                pressure = max(0.0, min(0.70, (0.85 - success_rate) * 1.4))
                shifted = transition[0] * pressure
                transition[0] -= shifted
                transition[1] += shifted * 0.35
                transition[2] += shifted * 0.65
            next_index = self._sample_index(transition)
            self.state = self.STATES[next_index]
        state_index = self.STATES.index(self.state)

        probability = config["attack_probability"][state_index]
        attacked = self.random.random() < probability
        attack_count = 0
        if attacked:
            rate = config["attack_lambda"][state_index]
            attack_count = min(self.max_attack, 1 + self._poisson(rate))

        return {
            "time": self.time,
            "method": "hmm",
            "state": self.state,
            "attack_probability": probability,
            "attacked": attacked,
            "attack_count": attack_count,
        }

    def _next_hmm(self, success_rate=None):
        return self._emit_hmm(advance_state=True, success_rate=success_rate)

    def _emit_ibal(self, success_rate, update_schedule):
        # IBAL increases P_max only when average learner success reaches eta.
        if (
            update_schedule
            and success_rate is not None
            and success_rate >= self.ibal_threshold
        ):
            self.p_max = min(1.0, self.ibal_alpha * self.p_max)

        lower = 1.0 / self.max_attack
        probability = self.random.uniform(lower, max(lower, self.p_max))
        group_size = self.random.randint(0, self.max_attack)
        attacked = group_size > 0 and self.random.random() < probability
        attack_count = group_size if attacked else 0
        self.state = "ibal"

        return {
            "time": self.time,
            "method": "ibal",
            "state": self.state,
            "attack_probability": probability,
            "p_max": self.p_max,
            "attacked": attacked,
            "attack_count": attack_count,
        }

    def _next_ibal(self, success_rate):
        return self._emit_ibal(success_rate, update_schedule=True)

    def nextstate(self, success_rate=None):
        """Update time and return the current numerical attack event."""
        self.time += 1
        if self.method == "ibal":
            return self._next_ibal(success_rate)
        return self._next_hmm(success_rate)

    def retry(self, success_rate=None):
        """Sample a retry attack without advancing the hidden attack stage."""
        primary_random = self.random
        self.random = self.retry_random
        try:
            if self.method == "ibal":
                return self._emit_ibal(success_rate, update_schedule=False)
            return self._emit_hmm(
                advance_state=False,
                success_rate=success_rate,
            )
        finally:
            self.random = primary_random


class multiattacker:
    """Independent attackers activated at different global time steps.

    Counts are conservatively added and capped at ``max_attack``. This models
    attackers compromising different carriers until every carrier is covered.
    """

    def __init__(
        self,
        start_times,
        seed=None,
        method="hmm",
        profile="bursty",
        max_attack=3,
    ):
        starts = tuple(int(value) for value in start_times)
        if not starts:
            raise ValueError("start_times cannot be empty")
        if any(value < 0 for value in starts):
            raise ValueError("start_times must be non-negative")
        self.start_times = starts
        self.max_attack = int(max_attack)
        self.time = 0
        base_seed = random.random() if seed is None else seed
        self.attackers = tuple(
            attacker(
                seed=f"{base_seed}:attacker:{index}",
                method=method,
                profile=profile,
                max_attack=max_attack,
                adaptive_pressure=True,
            )
            for index in range(len(starts))
        )

    def _inactive_event(self, index):
        return {
            "time": self.time,
            "method": "inactive",
            "state": "inactive",
            "attacked": False,
            "attack_count": 0,
            "attacker_index": index,
        }

    def _combine(self, events):
        attack_count = min(
            self.max_attack,
            sum(event["attack_count"] for event in events),
        )
        return {
            "time": self.time,
            "method": "multi",
            "state": tuple(event["state"] for event in events),
            "attacked": attack_count > 0,
            "attack_count": attack_count,
            "active_attackers": sum(
                self.time > start_time for start_time in self.start_times
            ),
            "events": events,
        }

    def nextstate(self, success_rate=None):
        self.time += 1
        events = []
        for index, (source, start_time) in enumerate(
            zip(self.attackers, self.start_times)
        ):
            if self.time <= start_time:
                events.append(self._inactive_event(index))
            else:
                event = source.nextstate(success_rate=success_rate)
                event["attacker_index"] = index
                events.append(event)
        return self._combine(events)

    def retry(self, success_rate=None):
        events = []
        for index, (source, start_time) in enumerate(
            zip(self.attackers, self.start_times)
        ):
            if self.time <= start_time:
                events.append(self._inactive_event(index))
            else:
                event = source.retry(success_rate=success_rate)
                event["attacker_index"] = index
                events.append(event)
        return self._combine(events)
