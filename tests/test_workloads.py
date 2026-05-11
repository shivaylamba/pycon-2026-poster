from pathlib import Path

from bench.workloads import load_rag_corpus, load_workload_items, rank_by_cosine


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_generation_workloads_load() -> None:
    assert load_workload_items("chat", DATA_DIR, limit=1)[0]["prompt"]
    assert load_workload_items("summarization", DATA_DIR, limit=1)[0]["prompt"]
    assert load_workload_items("classification", DATA_DIR, limit=1)[0]["expected"]


def test_rag_corpus_and_ranking() -> None:
    corpus = load_rag_corpus(DATA_DIR, limit=2)
    assert len(corpus) == 2
    ranked = rank_by_cosine([1.0, 0.0], [[0.8, 0.2], [0.1, 0.9]], top_k=1)
    assert ranked == [0]

