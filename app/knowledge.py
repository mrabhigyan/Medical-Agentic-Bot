
from functools import lru_cache

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


EMBEDDING_DIMENSION = 768


@lru_cache(maxsize=1)
def clients():
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing from your .env file.")

    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing from your .env file.")

    gemini_client = genai.Client(api_key=settings.gemini_api_key)
    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)

    return gemini_client, pinecone_client


def ensure_index():
    _, pc = clients()

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(settings.pinecone_index)


def embed(texts: list[str]) -> list[list[float]]:
    """Generate Gemini embeddings for a list of texts."""
    if not texts:
        return []

    gemini_client, _ = clients()
    embeddings = []

    # Process smaller batches to keep requests manageable.
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = gemini_client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=batch,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini did not return any embeddings.")

        embeddings.extend(
            item.values for item in response.embeddings
        )

    return embeddings


def search_kb(query: str, top_k: int = 5):
    """Search the medical knowledge base in Pinecone."""
    index = ensure_index()
    query_vector = embed([query])[0]

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = []

    for match in result.matches:
        if match.score is not None and match.score >= 0.30:
            metadata = match.metadata or {}

            matches.append({
                "page": metadata.get("page"),
                "text": metadata.get("text", ""),
                "score": round(match.score, 3),
                "source": metadata.get(
                    "file",
                    "Gale Encyclopedia (2002)",
                ),
            })

    return matches
