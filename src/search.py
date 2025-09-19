from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity for 1D vectors.
    Assumes inputs may not be normalized.
    """
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(an, bn))


def search_topk(
    query_embedding: np.ndarray,
    db_embeddings: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, scores) of top-k most similar entries.
    db_embeddings: (N, D) assumed normalized or not (we self-normalize here).
    """
    # Normalize both
    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    db = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-12)
    scores = db @ q  # cosine similarity since both are L2 normalized
    
    # Get top-k
    if len(scores) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    k = min(k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    # Sort descending
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores[idx]
