# Small RAG Pipeline

A small retrieval augmented generation pipeline has two distinct phases. First,
documents are embedded and stored in a vector index. This is a batch embedding
job and is usually measured as throughput: texts per second, tokens per second,
and energy per thousand tokens.

Second, each online query is embedded, compared with the stored vectors, and
used to select context for a generator. The ranking step may be tiny for a demo
corpus, but it becomes visible for larger collections or more complex rerankers.

