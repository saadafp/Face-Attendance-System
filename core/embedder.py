import numpy as np
import threading
from pathlib import Path
from insightface.model_zoo import get_model
import time

#region variables
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDER_PATH = BASE_DIR / "models" / "buffalo_m" / "w600k_r50.onnx"
_embedder = None
_lock = threading.Lock()
# endregion

def get_embedder():
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                t0 = time.perf_counter()
                _embedder = get_model(
                    name=str(EMBEDDER_PATH),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                t1 = time.perf_counter()
                print(f"embedder took {t1 - t0} seconds")
                _embedder.prepare(ctx_id=0, input_size=(112, 112))# GPU
    return _embedder


def embedder(blob):
    """
    This function uses a pre-trained face detector and creates embedding vectors.
    :param blob -> (1, 3, 112, 112)
    :return: norm_embedding -> (512,)
    """

    # forward to recognizer for embedding
    recognizer = get_embedder()
    embedding = recognizer.forward(blob)[0]

    # normalizing embedding
    norm = np.linalg.norm(embedding, ord=2)        #L2-norm
    norm_embedding = embedding / (norm + 1e-12)    # prevents dividing to 0
    return norm_embedding


