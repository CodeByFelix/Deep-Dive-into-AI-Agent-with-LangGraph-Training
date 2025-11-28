import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    """
    Simple RAG helper using SentenceTransformers + FAISS for embeddings/retrieval.
    Generation is left to the caller (e.g., OpenAI or any LLM).
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.id_to_text = {}

    def build_index(self, texts: List[str]):
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        # create FAISS index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        # store mapping
        self.id_to_text = {i: t for i, t in enumerate(texts)}

    def query(self, q: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        if self.index is None:
            raise RuntimeError('Index not built. Call build_index(texts) first.')
        q_emb = self.embedder.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((int(idx), float(score), self.id_to_text[int(idx)]))
        return results

