"""Lightweight, deterministic evaluation framework for the DibantuAI agent.

Replays scripted GLM trajectories through the real Agent loop and the
real registry business tools against the seeded mock store, so every
run is offline, free, and reproducible. See evaluation/README.md.

Only the dataset is re-exported here; import ``evaluation.evaluator``
directly (or run ``python -m evaluation.evaluator``) so the evaluator
module is not loaded twice.
"""

from .dataset import (
    CASES,
    CATEGORIES,
    EvalCase,
    GLMTurn,
    StateExpectation,
    ToolCall,
    load_cases,
)

__all__ = [
    "CASES",
    "CATEGORIES",
    "EvalCase",
    "GLMTurn",
    "StateExpectation",
    "ToolCall",
    "load_cases",
]
