"""
Vector store abstraction for the semantic half of hybrid retrieval.

    InMemoryVectorStore -- scores every receipt with the embedder (default).
    QdrantVectorStore   -- stores receipt vectors in Qdrant with the receipt
                           metadata as payload, and applies the APMC /
                           commodity / year / month filters inside Qdrant.

Both return {receipt_index: similarity} so the retriever can fuse the scores
with BM25 the same way.

Choose with the VECTOR_STORE environment variable (`memory` or `qdrant`).
Qdrant runs in local in-process mode by default; set QDRANT_URL
(e.g. http://localhost:6333) to use a Qdrant server instead.
"""
import os

COLLECTION = "apmc_receipts"


class InMemoryVectorStore:
    name = "memory"

    def __init__(self, embedder):
        self.embedder = embedder

    def scores(self, query: str, filters: dict) -> dict[int, float]:
        sims = self.embedder.similarity_scores(query)
        return {i: float(s) for i, s in enumerate(sims)}


def receipt_payload(receipt: dict) -> dict:
    year, month, _ = receipt["date"].split("-")
    return {
        "apmc": receipt["apmc"],
        "commodity": receipt["commodity"],
        "year": year,
        "month": month,
        "receipt_no": receipt["receipt_no"],
    }


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, embedder, documents: list[str], receipts: list[dict], url: str | None = None):
        from qdrant_client import QdrantClient, models  # noqa: local import

        self.embedder = embedder
        self.models = models
        url = url or os.environ.get("QDRANT_URL")
        self.client = QdrantClient(url=url) if url else QdrantClient(":memory:")

        vectors = embedder.embed(documents)
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        self.client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=i, vector=[float(x) for x in vec], payload=receipt_payload(r))
                for i, (vec, r) in enumerate(zip(vectors, receipts))
            ],
        )
        self.size = len(receipts)

    def _payload_filter(self, filters: dict):
        if not filters:
            return None
        return self.models.Filter(
            must=[
                self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

    def scores(self, query: str, filters: dict) -> dict[int, float]:
        query_vector = [float(x) for x in self.embedder.embed([query])[0]]
        points = self.client.query_points(
            COLLECTION,
            query=query_vector,
            query_filter=self._payload_filter(filters),
            limit=self.size,
        ).points
        return {int(p.id): float(p.score) for p in points}


def get_vector_store(embedder, documents, receipts, name: str | None = None):
    name = (name or os.environ.get("VECTOR_STORE", "memory")).lower()
    if name == "qdrant":
        return QdrantVectorStore(embedder, documents, receipts)
    return InMemoryVectorStore(embedder)
