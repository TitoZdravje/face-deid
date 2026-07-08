import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)

    if norm < eps:
        raise ValueError("Cannot normalize vector with near-zero norm.")

    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two embedding vectors.

    Higher means more similar.
    """

    a = l2_normalize(a)
    b = l2_normalize(b)

    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance based on cosine similarity.

    Lower means more similar.
    """

    return 1.0 - cosine_similarity(a, b)
