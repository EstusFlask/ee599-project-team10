"""Four defenders compared by ``train.py``.

The original class and method names are kept.  The baseline is fixed; the
other three models are a linear Q learner, a stateless lookup classifier, and
a history-aware k-nearest-neighbour classifier.
"""

import random
from collections import Counter


ACTIONS = (1, 3, 5, 7)


def defense_cost(times):
    """k=1 costs 0; every additional redundant pair costs one unit."""
    return (int(times) - 1) // 2


def defense_success(attack_count, times):
    """Majority is intact when fewer than ceil(k/2) copies are attacked."""
    return int(attack_count) < (int(times) + 1) // 2


def _history(observation, fallback=None):
    if isinstance(observation, dict):
        values = observation.get("history", fallback or [])
    elif isinstance(observation, (list, tuple)):
        values = observation
    elif observation is None:
        values = fallback or []
    else:
        values = [observation]
    return [int(value) for value in values]


def _allowed(observation, actions):
    if isinstance(observation, dict) and observation.get("allowed_actions"):
        return tuple(int(value) for value in observation["allowed_actions"])
    return tuple(actions)


def _nearest_action(action, allowed_actions):
    return min(allowed_actions, key=lambda value: (abs(value - action), value))


def _sequential_features(observation, window, fallback=None):
    """Identical observable inputs for stateful ML and RL."""
    values = _history(observation, fallback)[-window:]
    padded = [0] * (window - len(values)) + values
    short_values = padded[-min(5, window) :]
    short_mean = sum(short_values) / (3.0 * len(short_values))
    long_mean = sum(padded) / (3.0 * window)
    attack_rate = sum(value > 0 for value in padded) / window
    time_since_attack = window + 1
    for offset, value in enumerate(reversed(padded)):
        if value > 0:
            time_since_attack = offset
            break
    recency = min(1.0, time_since_attack / max(1, window))
    progress = 0.0
    remaining = 0.0
    message_length = 100.0
    importance = "normal"
    if isinstance(observation, dict):
        progress = float(observation.get("progress", 0.0))
        remaining = float(observation.get("remaining_budget", 0.0)) / 3.0
        message_length = float(observation.get("message_length", 100.0))
        importance = observation.get("importance", "normal")
    importance_value = {
        "low": 0.0,
        "normal": 0.5,
        "high": 1.0,
    }.get(importance, 0.5)
    risk = max(short_mean, long_mean, attack_rate)
    return (
        1.0,
        short_mean,
        long_mean,
        attack_rate,
        recency,
        progress,
        remaining,
        message_length / 100.0,
        importance_value,
        risk * progress,
        risk * remaining,
        progress * remaining,
    )


class RLdefender:
    def __init__(
        self,
        actions=ACTIONS,
        window=5,
        learning_rate=0.01,
        discount=0.99,
        epsilon=0.20,
        seed=0,
    ):
        self.states = []
        self.Q = {}
        self.actions = tuple(actions)
        self.window = int(window)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.random = random.Random(seed)
        self.weights = [[0.0] * 12 for _ in self.actions]
        self._last_state = None
        self._last_features = None
        self._last_action_index = None

    def reset(self):
        self.states = []
        self._last_state = None
        self._last_features = None
        self._last_action_index = None

    def observe(self, attack_count):
        self.states.append(int(attack_count))
        self.states = self.states[-self.window :]

    def _encode(self, observation=None):
        values = _history(observation, self.states)[-self.window :]
        padded = [0] * (self.window - len(values)) + values
        short_values = padded[-min(5, self.window) :]
        short_bucket = min(3, int(round(sum(short_values) / len(short_values))))
        long_bucket = min(3, int(round(sum(padded) / self.window)))
        time_since_attack = self.window + 1
        for offset, value in enumerate(reversed(padded)):
            if value > 0:
                time_since_attack = offset
                break
        if time_since_attack == 0:
            recency_bucket = 0
        elif time_since_attack <= 2:
            recency_bucket = 1
        elif time_since_attack <= 5:
            recency_bucket = 2
        else:
            recency_bucket = 3
        remaining = 0.0
        progress = 0.0
        importance = "normal"
        if isinstance(observation, dict):
            remaining = float(observation.get("remaining_budget", 0.0))
            progress = float(observation.get("progress", 0.0))
            importance = observation.get("importance", "normal")
        budget_bucket = max(0, min(3, int(round(remaining))))
        progress_bucket = max(0, min(3, int(progress * 4)))
        importance_bucket = {"low": 0, "normal": 1, "high": 2}.get(
            importance, 1
        )
        return (
            short_bucket,
            long_bucket,
            recency_bucket,
            budget_bucket,
            progress_bucket,
            importance_bucket,
        )

    def _features(self, observation=None):
        return _sequential_features(
            observation,
            self.window,
            fallback=self.states,
        )

    def _value(self, action_index, features):
        return sum(
            weight * feature
            for weight, feature in zip(self.weights[action_index], features)
        )

    def predict(self, observation=None, explore=False):
        """Decide the message repeat time from state and Q-table."""
        state = self._encode(observation)
        features = self._features(observation)
        values = [
            self._value(index, features) for index in range(len(self.actions))
        ]
        allowed_actions = _allowed(observation, self.actions)
        allowed_indices = [
            index for index, action in enumerate(self.actions)
            if action in allowed_actions
        ]
        if explore and self.random.random() < self.epsilon:
            action_index = self.random.choice(allowed_indices)
        else:
            best_value = max(values[index] for index in allowed_indices)
            best_indices = [
                index for index in allowed_indices if values[index] == best_value
            ]
            target_cost = 0.0
            if isinstance(observation, dict):
                target_cost = float(observation.get("remaining_budget", 0.0))
            action_index = min(
                best_indices,
                key=lambda index: (
                    abs(defense_cost(self.actions[index]) - target_cost),
                    self.actions[index],
                ),
            )
        self._last_state = state
        self._last_features = features
        self._last_action_index = action_index
        return self.actions[action_index]

    def update(self, state, reward, done=False):
        """Update the Q-table after observing reward and the next state."""
        if self._last_features is None or self._last_action_index is None:
            return
        next_features = self._features(state)
        allowed_actions = _allowed(state, self.actions)
        next_indices = [
            index for index, action in enumerate(self.actions)
            if action in allowed_actions
        ]
        future = 0.0 if done else max(
            self._value(index, next_features) for index in next_indices
        )
        target = float(reward) + self.discount * future
        old = self._value(self._last_action_index, self._last_features)
        error = max(-5.0, min(5.0, target - old))
        weights = self.weights[self._last_action_index]
        for index, feature in enumerate(self._last_features):
            weights[index] += self.learning_rate * error * feature


class fixdefender:
    def __init__(self, times):
        self.times = int(times)

    def predict(self, observation=None):
        allowed_actions = _allowed(observation, ACTIONS)
        return _nearest_action(self.times, allowed_actions)


class MLdefender_nostate:
    """Predict from current message length and importance, without history."""

    def __init__(self, actions=ACTIONS):
        self.actions = tuple(actions)
        self.action_by_context = {}
        self.default_action = self.actions[0]

    def _key(self, observation):
        if not isinstance(observation, dict):
            return 1, "normal"
        message_length = int(observation.get("message_length", 100))
        length_bucket = 0 if message_length <= 50 else 1 if message_length <= 100 else 2
        return length_bucket, observation.get("importance", "normal")

    def train(self, observations=None, labels=None):
        if observations is None or labels is None:
            raise ValueError("observations and labels are required")
        grouped = {}
        all_labels = []
        for observation, label in zip(observations, labels):
            key = self._key(observation)
            grouped.setdefault(key, []).append(int(label))
            all_labels.append(int(label))
        if all_labels:
            self.default_action = Counter(all_labels).most_common(1)[0][0]
        self.action_by_context = {
            key: Counter(values).most_common(1)[0][0]
            for key, values in grouped.items()
        }
        return self

    def predict(self, observation=None):
        action = self.action_by_context.get(self._key(observation), self.default_action)
        return _nearest_action(action, _allowed(observation, self.actions))


class MLdefender_withstate:
    """Small k-NN classifier using rolling attack-history features."""

    def __init__(self, actions=ACTIONS, window=5, neighbors=7, max_samples=3000):
        self.actions = tuple(actions)
        self.window = int(window)
        self.neighbors = int(neighbors)
        self.max_samples = int(max_samples)
        self.samples = []
        self.labels = []

    def _features(self, observation):
        return _sequential_features(observation, self.window)

    def train(self, observations=None, labels=None):
        if observations is None or labels is None:
            raise ValueError("observations and labels are required")
        pairs = list(zip(observations, labels))
        if len(pairs) > self.max_samples:
            stride = len(pairs) / self.max_samples
            pairs = [pairs[int(index * stride)] for index in range(self.max_samples)]
        self.samples = [self._features(observation) for observation, _ in pairs]
        self.labels = [int(label) for _, label in pairs]
        return self

    def predict(self, observation=None):
        if not self.samples:
            action = self.actions[0]
        else:
            target = self._features(observation)
            distances = []
            for sample, label in zip(self.samples, self.labels):
                distance = sum(
                    (left - right) ** 2 for left, right in zip(sample, target)
                )
                distances.append((distance, label))
            nearest = sorted(distances, key=lambda item: item[0])[: self.neighbors]
            action = Counter(label for _, label in nearest).most_common(1)[0][0]
        return _nearest_action(action, _allowed(observation, self.actions))
    
