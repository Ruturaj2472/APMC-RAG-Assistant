"""
Embedding provider abstraction.

Production path (requires internet + `pip install sentence-transformers`):
    BGEEmbedder -- uses BAAI/bge-small-en-v1.5 via sentence-transformers.

Offline/demo fallback (used automatically if sentence-transformers isn't
installed, e.g. in this sandbox):
    TfidfEmbedder -- scikit-learn TF-IDF vectors + cosine similarity.
    Same interface (`fit`, `embed`), so swapping providers requires no
    changes anywhere else in the pipeline.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfEmbedder:
    """Fallback embedder - no internet/model download required."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = None

    def fit(self, documents):
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        return self

    def similarity_scores(self, query: str):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]
        return sims


class BGEEmbedder:
    """
    Production embedder using Sentence Transformers (BAAI/bge-small-en-v1.5).
    Only imports the heavy dependency if actually instantiated, so the rest
    of the codebase works even when sentence-transformers isn't installed.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # noqa: local import
        import numpy as np  # noqa: local import

        self.model = SentenceTransformer(model_name)
        self.np = np
        self.doc_embeddings = None

    def fit(self, documents):
        self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)
        return self

    def similarity_scores(self, query: str):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        return self.doc_embeddings @ q_emb


def get_embedder():
    """Return the best available embedder: BGE if installed, else TF-IDF fallback."""
    try:
        return BGEEmbedder()
    except ImportError:
        print("[embeddings] sentence-transformers not available - using TF-IDF fallback.")
        return TfidfEmbedder()
