from src.abstention import should_abstain
from src.chunker import Chunk
from src.retrieval.hybrid import Hit


def _h(score):
    c = Chunk("d", f"d:{score}", "t", 0, 1)
    return Hit(chunk=c, score=score)


def test_no_hits_abstains():
    assert should_abstain([], {"mode": "score_and_gap"}) is True


def test_above_threshold_does_not_abstain():
    hits = [_h(0.8), _h(0.5)]
    assert should_abstain(hits, {"mode": "score", "min_top_score": 0.3}) is False


def test_below_threshold_abstains():
    hits = [_h(0.1)]
    assert should_abstain(hits, {"mode": "score", "min_top_score": 0.3}) is True
