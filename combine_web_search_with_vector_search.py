import os
from typing import Any, cast

import tiktoken
from openai import OpenAI
from tavily import TavilyClient

from fixed_length_chunking import fixed_length_chunking
from get_embeddings import get_embeddings
from vector_search import vector_search

if __name__ == "__main__":
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    query: str = "2025 Nobel Prize winners"
    max_results: int = 10
    include_raw_content: bool = True

    # STEP 1: web search
    response: dict[str, Any] = cast(
        dict[str, Any],
        tavily.search(  # pyright: ignore[reportUnknownMemberType]
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
        ),
    )

    search_results = []
    for result in response["results"]:
        if result.get("raw_content"):
            search_results.append(
                {
                    "title": result["title"],
                    "content": result["raw_content"],
                    "url": result["url"],
                }
            )

    # STEP 2: check token count
    enc = tiktoken.encoding_for_model("gpt-5")
    full_text = "\n\n".join([f"Title: {r['title']}\n{r['content']}" for r in search_results])
    total_tokens = len(enc.encode(full_text))

    print(f"Length of full text: {len(full_text)}")
    print(f"Total tokens: {total_tokens}")

    # STEP 3: chunking and embedding
    all_chunks: list[dict[str, Any]] = []
    for result in search_results:
        text = f"Title: {result['title']}\n{result['content']}"
        chunks = fixed_length_chunking(text=text, chunk_size=500, overlap=50)
        for chunk in chunks:
            all_chunks.append(
                {
                    "text": chunk,
                    "title": result["title"],
                    "url": result["url"],
                }
            )

    print(f"Total chunks: {len(all_chunks)}")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = get_embeddings(client=client, texts=chunk_texts)
    # just print the first 5 embeddings
    for i in range(5):
        print(f"Embedding {i}: {chunk_embeddings[i]}")

    # STEP 4: execute vector search
    query_in_context: str = "quantum computing"
    results = vector_search(
        client=client,
        query=query_in_context,
        chunks=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
    )

    print(f"Query: '{query_in_context}'\n")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {r['similarity']:.3f}")
        print(f"{r['chunk'][:300]}...")

    # STEP 5: token savings effect
    top_chunks = [r["chunk"] for r in results]
    selected_text = "\n\n".join(top_chunks)
    selected_tokens = len(enc.encode(selected_text))

    print(f"Total tokens: {total_tokens}")
    print(f"Selected tokens: {selected_tokens}")
    print(f"Savings rate: {(1 - selected_tokens / total_tokens) * 100:.1f}%")
