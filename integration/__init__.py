"""Adaptive-redundancy defense on top of the existing BFT engine (no source files modified).

The RL / rule policy picks the number of channels ``k`` for every message; the real
``core.engine.DefenseEngine`` does the hashing and voting; a mock adversary replaces the LLM.
"""
