# Face Recognition Attendance System

Real-time attendance system that recognizes faces from a webcam or a video file, logs each person once to a database, and shows the live feed and recognized names in the browser.

<!-- Add a screenshot or GIF here: docs/demo.gif -->
<!-- ![Demo](docs/demo.gif) -->

## How it works

```
Frame → Face Detection → Landmark Alignment → Embedding → Cosine Matching → Temporal Verification → SQLite
```

1. **Detection**: SCRFD (`det_2.5g`, InsightFace `buffalo_m`) returns bounding boxes and 5 facial landmarks.
2. **Alignment**: `norm_crop` warps each face to a canonical 112×112 crop using the landmarks.
3. **Embedding**: ArcFace (`w600k_r50`) produces a 512-d vector, L2-normalized.
4. **Matching**: cosine similarity (a plain dot product, since vectors are normalized) against one mean embedding per known person. Below the threshold (default `0.4`) the face is labeled "unknown".
5. **Temporal verification**: a person is registered only if matched at least `MIN_HITS` times (default 3) within `TIME_WINDOW_SECONDS` (default 3 s). This suppresses single-frame false positives.
6. **Persistence**: registered people are written to `PresenceFaces.db` (name, score, timestamp). Each person is registered once per run.

The web layer is Flask. A background `CameraWorker` thread runs the pipeline and shares the latest frame and state with the request handlers under a lock, so inference never blocks HTTP requests.

## Features

- Multi-face detection and recognition per frame
- Webcam (`source=0`) or video file input
- Live MJPEG stream in the browser (`/video_feed`)
- JSON endpoints for current faces (`/current_faces`) and new registrations (`/events`)
- Incremental enrollment: only new people in the dataset folder are embedded
- GPU inference via ONNX Runtime (CUDA) with automatic CPU fallback
- Per-stage timing logs (detect / align + embed + match)

## Tech stack

Python · InsightFace · ONNX Runtime · OpenCV · NumPy · SQLite · Flask

## Project structure

```
.
├── app.py                      # Flask app, MJPEG stream, JSON endpoints
├── core/
│   ├── detector.py             # SCRFD face detector (lazy, thread-safe singleton)
│   ├── aligner.py              # landmark-based alignment -> (1,3,112,112) blob
│   ├── embedder.py             # ArcFace embedding + L2 normalization
│   └── matcher.py              # cosine-similarity matching
├── database/
│   ├── database.py             # SQLite helpers (embeddings DB, presence DB)
│   └── add_embeddings_to_db.py # enroll new people from the dataset folder
├── service/
│   ├── presence.py             # capture + recognition + verification loop (generator)
│   ├── camera_worker.py        # background thread wrapper for the Flask app
├── templates/
│   └── index.html
├── models/buffalo_m/           # NOT included (see Setup)
└── data/known_faces/           # NOT included (see Setup)
```

## Setup

### 1. Download the models

Download the InsightFace `buffalo_m` pack and place these two files in `models/buffalo_m/`:

- `det_2.5g.onnx`
- `w600k_r50.onnx`

### 2. Add known faces

Create one folder per person. The folder name is the label shown in the UI. Use several clear, front-facing photos per person.

```
data/known_faces/Celebrity_Faces_Dataset/
├── Person One/
│   ├── 1.jpg
│   └── 2.jpg
└── Person Two/
    └── 1.jpg
```

### 3. Enroll

```bash
python -m service.add_embeddings_to_db
```

Running it again later only embeds people who are not yet in the database.

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

To switch the input source, change the `source` passed to `CameraWorker` in `app.py`:

```python
# webcam
worker = camera_worker.CameraWorker(source=0)

# video file
worker = camera_worker.CameraWorker(source="path/to/video.mp4")
```

## Configuration

Defined at the top of `service/presence.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `THRESHOLD` | `0.4` | Minimum cosine similarity to accept a match |
| `MIN_HITS` | `3` | Detections required before registering |
| `TIME_WINDOW_SECONDS` | `3.0` | Window in which those hits must occur |
| `MAX_PROCESS_FPS` | `5` | Cap on processed frames per second |

## API

| Endpoint | Returns |
|---|---|
| `/` | Live dashboard |
| `/video_feed` | MJPEG stream |
| `/current_faces` | Faces in the latest frame: `bbox`, `name`, `status` (`unknown` / `verifying` / `registered`), `score` |
| `/events` | New registrations: `name`, `score`, `time` |

## Limitations

- No liveness / anti-spoofing: a printed photo or a screen can be recognized.
- One mean embedding per person is a single prototype, so it is weaker than per-image matching under large pose or lighting changes.
- Matching is a brute-force dot product: fine for hundreds of identities, not for very large galleries.


## Roadmap

- [ ] Liveness detection
- [ ] Daily attendance reset and per-day records
- [ ] Attendance report / CSV export
- [ ] Cross-platform webcam backend
- [ ] Docker image

## Acknowledgements

[InsightFace](https://github.com/deepinsight/insightface) for the SCRFD and ArcFace models. Check their license terms before any commercial use.
