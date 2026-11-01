from src.chunker import chunk_fixed_tokens, chunk_sentence


def test_fixed_chunks_cover_document():
    text = "a" * 1200
    chunks = chunk_fixed_tokens("d", text, size=500, overlap=100)
    # offsets reconstruct the original text
    assert chunks[0].start == 0
    assert chunks[-1].end == 1200
    for c in chunks:
        assert c.text == text[c.start:c.end]


def test_fixed_chunks_overlap_is_correct():
    text = "a" * 1000
    chunks = chunk_fixed_tokens("d", text, size=400, overlap=50)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == a.end - 50


def test_sentence_chunks_respect_bounds():
    text = "Alpha sentence. Beta sentence. Gamma sentence. " * 10
    chunks = chunk_sentence("d", text, size=120, overlap=0)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for c in chunks:
        assert c.text == text[c.start:c.end]
