"""Adapter that lets a controller pick ``k`` before every transmission.

Subclasses ``BaseDefenseAdapter`` and reuses its ``secure_transmit`` (packet creation,
adversary hook, ``engine.resolve``).  Works with any framework adapter that calls
``secure_transmit`` (AutoGen / CAMEL / MetaGPT), since only this method is wrapped.
"""

from __future__ import annotations

from adapters.base_adapters import BaseDefenseAdapter


class AdaptiveAdapter(BaseDefenseAdapter):
    def __init__(self, engine, adversary, protocols, victim_name, controller,
                 attack_model, importance=lambda: 1.0, logger=None) -> None:
        """``attack_model(k) -> list of logical channel ids`` the adversary controls this
        round (a deployment has no such hook; it exists to drive the simulation).
        ``importance()`` returns the importance of the message about to be sent."""
        super().__init__(engine, adversary, protocols, victim_name, logger)
        self.controller = controller
        self.attack_model = attack_model
        self.importance = importance
        self.history: list[dict] = []

    def secure_transmit(self, message, sender_name, receiver_name):
        message = str(message).strip()
        length = len(message.encode())
        importance = self.importance()
        k = self.controller.decide(length, importance)
        self.engine.set_k(k)
        self.attacked_channels = list(self.attack_model(k))
        final_message, traitors = super().secure_transmit(message, sender_name, receiver_name)
        # Signed receipt: the sender compares the winning hash with the hash of what it sent.
        verified = self.engine.integrity_verified()
        self.controller.feedback(k, len(traitors), verified, self.engine.total_bytes, length)
        self.history.append({
            "k": k, "attacked": len(self.attacked_channels), "dissent": len(traitors),
            "verified": verified, "bytes": self.engine.total_bytes,
            "length": length, "importance": importance,
        })
        return final_message, traitors
