# region importing libraries
import cv2
import time
import datetime
import threading
import logging
import sys
from collections import deque
from core.detector import detect_faces
from core.aligner import align_face
from core.embedder import embedder
from database.database import *
from core.matcher import match_face
from collections import defaultdict
from pathlib import Path
# endregion

# region variables
BASE_DIR = Path(__file__).resolve().parent.parent
CELEB_DB_PATH = BASE_DIR / "database" / "Celeb_face_embeddings.db"
PRESENCE_DB_PATH = BASE_DIR / "database" / "PresenceFaces.db"
DATASET_PATH = BASE_DIR / "data" / "known_faces" / "Celebrity_Faces_Dataset"
VIDEO_PATH = BASE_DIR / "data" / "inputs" / "taylor swift.MP4"
MIN_HITS = 3  # seen at least 3 times
THRESHOLD = 0.4
TIME_WINDOW_SECONDS = 3.0
MAX_PROCESS_FPS = 5
FRAME_SKIP = 30
# endregion

#region logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("attendance.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
#endregion

#region run webcam
def run_presence(source, celeb_faces_db_path=CELEB_DB_PATH, presence_faces_db_path=PRESENCE_DB_PATH,
                  threshold=THRESHOLD, min_hits=MIN_HITS, time_window_seconds=TIME_WINDOW_SECONDS,
                  max_process_fps=MAX_PROCESS_FPS, stop_event: threading.Event = None):
    """
    Processes webcam frames and registers a person only if their face is detected
    at least `min_hits` times within `time_window_seconds` continuously.
    """
    # checks if the source
    is_webcam = isinstance(source, int)
    # newest frame for webcam, larger buffer for video to smooth the process
    cap = cv2.VideoCapture(source if is_webcam else str(source), cv2.CAP_DSHOW if is_webcam else 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1 if is_webcam else 3)

    if not cap.isOpened():
        kind = "webcam index" if is_webcam else "file"
        raise RuntimeError(f"Cannot open {kind}: {source}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source opened | webcam={is_webcam} | fps={actual_fps} | resolution={width}x{height}")

    # accessing databases
    db_names, db_embeddings = access_db(db_path=celeb_faces_db_path)
    conn = presence_db(db_path=presence_faces_db_path)

    # data structures for time-based logic
    detection_timestamps = defaultdict(lambda: deque(maxlen=10))
    registered = set()

    min_interval = 1.0 / max_process_fps
    last_processed = 0.0
    frame_id = 0

    try:
        while cap.isOpened():
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop signal received")
                break

            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            if now - last_processed < min_interval:
                continue
            last_processed = now
            frame_id += 1

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                t_start = time.perf_counter()

                # multi-face detection
                faces = detect_faces(rgb_frame)
                t_detect = time.perf_counter()


                current_time = time.perf_counter()
                frame_faces = []
                new_registration = None

                # multi-face recognition
                for bbox, lm in faces:
                    blob = align_face(rgb_frame, lm, image_size=112)
                    norm_embedding = embedder(blob)
                    name, score = match_face(norm_embedding, db_embeddings, db_names, threshold=threshold)

                    if name is None :
                        status = "unknown"
                    elif name in registered:
                        status = "registered"
                    else:
                        status = "verifying"

                        detection_timestamps[name].append(current_time)
                        timestamps = list(detection_timestamps[name])

                        if len(timestamps) >= min_hits:
                            time_span = timestamps[-1] - timestamps[-min_hits]
                            if time_span <= time_window_seconds:
                                date = datetime.datetime.now()
                                add_to_presence_db(conn, name, score, date)
                                registered.add(name)
                                status = "registered"
                                logger.info(f"REGISTERED: {name} | similarity={score:.3f}")
                                new_registration = {"name": name, "score": float(score), "time": date.isoformat()}

                    frame_faces.append({
                        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        "name": name,
                        "status": status,
                        "score": float(score)})


                t_end = time.perf_counter()
                logger.info(
                    f"Frame {frame_id} timing | "
                    f"detect={(t_detect - t_start) * 1000:.1f}ms | "
                    f"align+embed+match={(t_end - t_detect) * 1000:.1f}ms | "
                    f"total={(t_end - t_start) * 1000:.1f}ms | "
                    f"faces={len(faces)}"
                )

                yield {
                "frame": frame,
                "faces": frame_faces,
                "new_registration": new_registration}

            except Exception as e:
                logger.warning(f"Frame {frame_id} error: {e}")
                continue

    finally:
        cap.release()
        conn.close()
        logger.info(f"Stream closed | total processed frames={frame_id}")

#endregion

#region run
if __name__ == "__main__":
    stop_event = threading.Event()
    try:
        for event in run_presence(source=str(VIDEO_PATH), stop_event=stop_event):
            statuses = [f["status"] for f in event["faces"]]
            print(f"faces={len(event['faces'])} | statuses={statuses}")
            if event["new_registration"]:
                print("NEW REGISTRATION:", event["new_registration"])
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping...")
#endregion