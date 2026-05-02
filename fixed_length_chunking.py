def fixed_length_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-length chunks."""

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end: int = start + chunk_size
        chunk: str = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end

    return chunks


if __name__ == "__main__":
    sample = "A" * 283
    chunks = fixed_length_chunking(text=sample, chunk_size=100, overlap=20)
    print(f"Original: {len(sample)} chars -> {len(chunks)} chunks")
