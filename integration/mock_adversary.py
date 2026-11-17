"""LLM-free adversary: same interface as ``agents.adversary.LlamaAdversary.manipulate``."""

from __future__ import annotations


class MockAdversary:
    """Rewrites a message deterministically (all compromised channels send the same poison,
    as in ``adapters.base_adapters`` where the poison is generated once and cached)."""

    def __init__(self, marker: str = "[TAMPERED]") -> None:
        self.marker = marker
        self.calls = 0

    def manipulate(self, message: str, sender_name: str, receiver_name: str) -> str:
        self.calls += 1
        return f"{self.marker}{message}"
