import os
from typing import Any

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from get_embeddings import get_embeddings


def vector_search(
    client: OpenAI, query: str, chunks: list[str], chunk_embeddings: np.ndarray, top_k: int = 3
) -> list[dict[str, Any]]:
    """Find the most similar chunks to the query."""

    query_embedding = get_embeddings(client=client, texts=[query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for idx in top_indices:
        results.append({"chunk": chunks[idx], "similarity": similarities[idx]})

    return results


if __name__ == "__main__":
    documents = [
        "Python is a programming language",
        "Machine learning uses Python extensively",
        "Cats are popular pets",
        "Deep learning is a subset of machine learning",
    ]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    doc_embeddings = get_embeddings(client=client, texts=documents)

    results = vector_search(
        client=client,
        query="Artificial intelligence",
        chunks=documents,
        chunk_embeddings=doc_embeddings,
        top_k=4,
    )

    for r in results:
        print(f"Similarity: {r['similarity']:.3f}: {r['chunk']}")
