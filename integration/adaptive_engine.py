"""``DefenseEngine`` with a per-message channel count, byte accounting and a hash check."""

from __future__ import annotations

from collections import Counter

from core.engine import DefenseEngine

HASH_BYTES = 64  # SHA-256 hex digest sent by witness channels


class AdaptiveDefenseEngine(DefenseEngine):
    """Same hashing and majority vote as the base engine; only ``k`` is dynamic.

    A round is a success when the hash that wins the vote is the hash of the message the
    sender transmitted (``integrity_verified``).  Then the receiver can verify any received
    text against the winning hash and knows whether it was modified; this is the "fewer than
    half of the channels altered" rule of the simulator.  It does not require the original
    text to arrive: only 2 channels carry full text.
    """

    def __init__(self, num_text_carriers: int = 2) -> None:
        super().__init__(k=1, num_text_carriers=num_text_carriers, max_ghost_channels=0)
        self.last_bytes = 0
        self._original_hash = ""
        self._winner_hash = None
        self._winner_votes = 0

    def set_k(self, k: int) -> None:
        if k < 1 or k % 2 == 0:
            raise ValueError("k must be a positive odd number")
        self.k = k
        self.active_channel_ids = list(range(k))

    def prepare_packets(self, message):
        packets = super().prepare_packets(message)
        self._original_hash = self._get_hash(message)
        self._winner_hash, self._winner_votes = None, 0
        self.last_bytes = sum(
            len(str(p["content"]).encode()) if p["type"] == "text" else HASH_BYTES
            for p in packets
        )
        return packets

    def resolve(self, candidates):
        votes = Counter(c.get("hash", "DROPPED") for c in candidates)
        self._winner_hash, self._winner_votes = votes.most_common(1)[0]
        return super().resolve(candidates)

    def integrity_verified(self) -> bool:
        """True when a strict majority of channels voted for the sender's original hash."""
        return (self._winner_hash == self._original_hash
                and self._winner_votes > self.k / 2)

    @property
    def total_bytes(self) -> int:
        return self.last_bytes
