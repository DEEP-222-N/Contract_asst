"""Decide when the system should return nothing."""
from __future__ import annotations

from .retrieval.hybrid import Hit


def should_abstain(hits: list[Hit], cfg: dict) -> bool:
    mode = cfg.get("mode", "score_and_gap")
    if mode == "none" or not hits:
        return not hits  # no hits at all => abstain
    top = hits[0].score
    if mode in ("score", "score_and_gap"):
        if top < float(cfg.get("min_top_score", 0.0)):
            return True
    if mode == "score_and_gap" and len(hits) >= 2:
        second = hits[1].score
        if (top - second) < float(cfg.get("min_gap", 0.0)):
            # top is weak and crowded — more likely spurious
            if top < 2 * float(cfg.get("min_top_score", 0.0)):
                return True
    return False
