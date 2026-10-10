#region import
import cv2
import time
import signal
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, Response
from service import camera_worker
#endregion

#region logging
logger = logging.getLogger(__name__)
#endregion

#region variables
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "inputs" / "taylor swift.MP4"
#endregion

#region configurations
app = Flask(__name__)
worker = camera_worker.CameraWorker(source=str(VIDEO_PATH))
worker.start()
#endregion


#region functions
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return "This is An Attendance System!"

@app.route("/events")
def events():
    return jsonify(worker.get_events())

@app.route("/current_faces")
def current_faces():
    return jsonify(worker.get_current_faces())

def generate_mjpeg():
    target_fps = 15
    min_interval = 1.0 / target_fps

    while True:
        start = time.perf_counter()
        frame = worker.get_latest_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        elapsed = time.perf_counter() - start
        sleep_time = min_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")
#endregion

#region run
if __name__ == "__main__":
    app.run(debug=False, threaded=True)
#endregion