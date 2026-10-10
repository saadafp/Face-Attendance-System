import numpy as np

def match_face(query_embedding: np.ndarray, db_embeddings: np.ndarray,db_names: list,threshold: float = 0.4):
    """
    Returns:
        best_name (str or None)
        best_score (float)
    """
    if db_embeddings is None or len(db_embeddings) == 0:
        return None, 0.0

    # cosine similarity (embeddings are L2-normalized)
    similarities = np.dot(db_embeddings, query_embedding) # [-1, 1] the more score, more similarity

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return db_names[best_idx], float(best_score)
    else:
        return None, float(best_score)


