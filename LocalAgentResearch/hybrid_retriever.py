"""
Hybrid Retriever: Dense (BGE) + Sparse (BM25) with Alpha-Weighted
Reciprocal Rank Fusion (RRF).

Replaces vectorstore.similarity_search_with_score() across all pipelines.
Score convention: LOWER = BETTER (negated RRF), compatible with all
downstream code (sort ascending).

Supports both filtered (HotpotQA) and open-domain retrieval.
Alpha: 0.0 = pure BM25, 1.0 = pure semantic (dense).

BM25 index is cached to disk (pickle) to avoid rebuilding on every run.
"""

import hashlib
import os
import pickle
import yaml
from collections import defaultdict
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Globals — built once per process, persisted to disk via pickle
# ---------------------------------------------------------------------------
_bm25: BM25Okapi | None = None
_bm25_by_qid: dict[str, list[int]] | None = None
# {question_id: [bm25_index, ...]}  — efficient batch scoring in filtered mode

_hash_to_bm25_idx: dict[str, int] | None = None
# {md5_hex(content): bm25_index}  — stable content→index mapping for alignment

BM25_CACHE_PATH = "./BM25_CACHE/bm25_index.pkl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_alpha() -> float:
    """Read hybrid_alpha from config.yaml, default to 0.5 if missing."""
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get("database", {}).get("hybrid_alpha", 0.5)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip trailing punctuation, preserve hyphens."""
    if not text:
        return []
    return [word.strip(".,!?()[]{}\"'") for word in text.lower().split()]


def _md5(text: str) -> str:
    """Stable content hash — survives pickle round-trips across sessions."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _build_bm25(vectorstore) -> None:
    """Build BM25 index from all ChromaDB documents and cache to disk."""
    global _bm25, _bm25_by_qid, _hash_to_bm25_idx

    print("Building global BM25 index from vectorstore...")
    total = vectorstore._collection.count()
    batch_size = 500

    corpus: list[list[str]] = []
    _bm25_by_qid = defaultdict(list)
    _hash_to_bm25_idx = {}

    for offset in range(0, total, batch_size):
        data = vectorstore.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for content, meta in zip(data["documents"], data["metadatas"]):
            idx = len(corpus)
            tokens = _tokenize(content)
            corpus.append(tokens)
            qid = meta.get("question_id", "")
            _bm25_by_qid[qid].append(idx)
            _hash_to_bm25_idx[_md5(content)] = idx

    _bm25 = BM25Okapi(corpus)

    # Persist to disk
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(
            {"bm25": _bm25, "hash_map": _hash_to_bm25_idx, "qid_map": _bm25_by_qid}, f
        )

    print(
        f"BM25 index ready: {len(corpus)} docs across {len(_bm25_by_qid)} questions."
    )


def _load_or_build_bm25(vectorstore) -> None:
    """Load cached BM25 from disk, or build if missing."""
    global _bm25, _bm25_by_qid, _hash_to_bm25_idx

    if os.path.exists(BM25_CACHE_PATH):
        print("Loading pre-built BM25 index from disk...")
        with open(BM25_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
            _bm25 = cache["bm25"]
            _hash_to_bm25_idx = cache["hash_map"]
            _bm25_by_qid = cache["qid_map"]
    else:
        _build_bm25(vectorstore)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_search_with_score(
        query: str,
        question_id: str | None,
        vectorstore,
        k: int,
        alpha: float | None = None,
) -> list[tuple]:
    """
    Hybrid dense + BM25 retrieval via alpha-weighted Reciprocal Rank Fusion.

    Args:
        query:        Natural language query string.
        question_id:  HotpotQA question ID for filtered retrieval.
                      Pass None for open-domain (no filter).
        vectorstore:  ChromaDB vectorstore instance.
        k:            Number of documents to return.
        alpha:        Fusion weight (0 = pure BM25, 1 = pure dense).
                      Reads from config.yaml if None.

    Returns:
        List of (Document, hybrid_score) tuples.
        hybrid_score: LOWER = BETTER (negated RRF).
    """
    global _bm25, _bm25_by_qid, _hash_to_bm25_idx

    if alpha is None:
        alpha = _load_alpha()
    alpha = max(0.0, min(1.0, alpha))

    # --- Lazy init ---
    if _bm25 is None:
        _load_or_build_bm25(vectorstore)

    # --- 1. Dense retrieval ---
    search_kwargs: dict = {"k": max(k * 2, 20)}
    if question_id is not None:
        search_kwargs["filter"] = {"question_id": question_id}

    raw = vectorstore.similarity_search_with_score(query, **search_kwargs)
    docs: list = [doc for doc, _ in raw]
    dense_dists: list[float] = [score for _, score in raw]

    # --- 2. Sparse (BM25) scores ---
    query_tokens = _tokenize(query)
    bm25_sims: list[float] = []

    if question_id is not None:
        # Filtered mode: score all docs for this question, align by content hash
        bm25_indices = _bm25_by_qid.get(question_id, [])
        idx_to_score = dict(
            zip(bm25_indices, _bm25.get_batch_scores(query_tokens, bm25_indices))
        )
        for doc in docs:
            bm25_idx = _hash_to_bm25_idx.get(_md5(doc.page_content))
            bm25_sims.append(
                idx_to_score.get(bm25_idx, 0.0) if bm25_idx is not None else 0.0
            )
    else:
        # Open-domain mode: look up each dense doc's BM25 index via content hash
        indices_to_score: list[int] = []
        doc_idx_map: dict[int, int] = {}  # bm25_idx → position in docs
        for i, doc in enumerate(docs):
            bm25_idx = _hash_to_bm25_idx.get(_md5(doc.page_content))
            if bm25_idx is not None:
                indices_to_score.append(bm25_idx)
                doc_idx_map[bm25_idx] = i

        scores = _bm25.get_batch_scores(query_tokens, indices_to_score)
        bm25_sims = [0.0] * len(docs)
        for bm25_idx, sim in zip(indices_to_score, scores):
            bm25_sims[doc_idx_map[bm25_idx]] = sim

    # --- 3. Alpha-weighted RRF ---
    k_rrf = 60
    penalty_rank = max(len(docs), 10) + 1  # for docs absent from one retriever

    # Dense rank: lower distance → rank 1
    dense_order = sorted(range(len(dense_dists)), key=lambda i: dense_dists[i])
    dense_rank = [0] * len(dense_dists)
    for rank, idx in enumerate(dense_order):
        dense_rank[idx] = rank + 1

    # BM25 rank: higher similarity → rank 1 (penalty rank for zero-score docs)
    bm25_order = sorted(range(len(bm25_sims)), key=lambda i: bm25_sims[i], reverse=True)
    bm25_rank = [penalty_rank] * len(bm25_sims)
    for rank, idx in enumerate(bm25_order):
        bm25_rank[idx] = rank + 1

    # RRF: higher = better. Negate → lower = better.
    hybrid = [
        -(
                alpha * 1.0 / (k_rrf + dense_rank[i])
                + (1.0 - alpha) * 1.0 / (k_rrf + bm25_rank[i])
        )
        for i in range(len(docs))
    ]

    # --- 4. Sort ascending, return top-k ---
    combined = list(zip(docs, hybrid))
    combined.sort(key=lambda x: x[1])
    return combined[:k]
