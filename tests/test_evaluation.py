from src.chunker import Chunk
from src.evaluation import char_precision_recall
from src.retrieval.hybrid import Hit


def _hit(doc, start, end, score=1.0):
    c = Chunk(doc_id=doc, chunk_id=f"{doc}:{start}-{end}", text="x" * (end - start),
              start=start, end=end)
    return Hit(chunk=c, score=score)


def test_perfect_match_gives_1_0():
    hits = [_hit("d1", 10, 50)]
    gt = [("d1", 10, 50)]
    p, r = char_precision_recall(hits, gt, k=1)
    assert p == 1.0 and r == 1.0


def test_partial_cover():
    hits = [_hit("d1", 10, 30)]  # 20 chars
    gt = [("d1", 10, 50)]        # 40 chars
    p, r = char_precision_recall(hits, gt, k=1)
    assert p == 1.0
    assert r == 0.5


def test_wrong_doc_zero_overlap():
    hits = [_hit("d2", 10, 50)]
    gt = [("d1", 10, 50)]
    p, r = char_precision_recall(hits, gt, k=1)
    assert p == 0.0 and r == 0.0


def test_merge_overlapping_retrieved_intervals():
    hits = [_hit("d1", 10, 30), _hit("d1", 20, 40)]  # merged: [10,40) => 30 chars
    gt = [("d1", 10, 50)]                             # 40 chars
    p, r = char_precision_recall(hits, gt, k=2)
    assert p == 1.0
    assert abs(r - 30 / 40) < 1e-9
