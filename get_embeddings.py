import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(client,texts, model="text-embedding-3-small"):
    """Convert text to embedding vectors."""

    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(input=texts, model=model)

    return np.array([item.embedding for item in response.data])



def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sentences = [
        "The cat is sleeping on the couch",
        "A kitten is playing with a toy",
        "The dog is running in the park",
    ]
    embeddings = get_embeddings(client=client,texts=sentences)

    cat_kitten = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    cat_dog = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

    print(f"Cat vs kitten: {cat_kitten:.3f}")
    print(f"Cat vs Dog: {cat_dog:.3f}")


if __name__ == "__main__":
    main()
    exit(0)
