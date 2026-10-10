import threading
from pathlib import Path
from insightface.model_zoo import get_model
import time


#region variables
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH = BASE_DIR / "models" / "buffalo_m" / "det_2.5g.onnx"
_detector = None
_lock = threading.Lock()
# endregion

def get_detector():
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                t0 = time.perf_counter()
                _detector = get_model(
                    name=str(DETECTOR_PATH),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                _detector.prepare(ctx_id=0, input_size=(640, 640))
                t1 = time.perf_counter()
                print(f"Detector took {t1 - t0} seconds")
    return _detector
def detect_faces(image, conf_threshold=0.6):
    """
    Lazily initializes and returns the face detector model.
    Why this exists:
    - Prevents model loading during module import
    - Avoids path issues during deployment
    - Ensures a single shared model instance is reused
    - Makes the code safe for FastAPI, Docker, and production servers

    This function is used to detect faces from a video/image file.
    :param image
    :param conf_threshold
    :return: bboxes, landmarks --> (bbox + 5 landmarks)
    """
    # preparing detector model
    # detecting bboxes and landmarks
    detector = get_detector()
    bboxes, landmarks = detector.detect(image)
    # filter by confidence threshold
    faces = [(b, l) for b, l in zip(bboxes, landmarks) if b[4] > conf_threshold]

    return faces



