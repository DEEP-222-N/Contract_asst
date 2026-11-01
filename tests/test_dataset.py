from src.dataset import assign_split, qid_for, mini


def test_split_is_deterministic():
    s1 = assign_split("Does this agreement allow sublicensing?")
    s2 = assign_split("Does this agreement allow sublicensing?")
    assert s1 == s2
    assert s1 in ("train", "test")


def test_qid_is_stable_and_short():
    q = "What is the term?"
    assert qid_for(q) == qid_for(q)
    assert len(qid_for(q)) == 12


def test_mini_deterministic_subset():
    tests = [{"snippets": [{"file_path": f"doc{i}.txt"}]} for i in range(300)]
    a = mini(tests, n=50)
    b = mini(tests, n=50)
    assert len(a) == 50
    assert [t["snippets"][0]["file_path"] for t in a] == \
           [t["snippets"][0]["file_path"] for t in b]
