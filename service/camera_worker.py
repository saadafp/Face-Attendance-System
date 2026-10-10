# camera_worker.py
import threading
import logging
from service.presence import run_presence

logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, source=0):
        self.source = source
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.current_faces = []
        self.events = []

        self.thread = None

    def _run(self):
        try:
            for state in run_presence(source=self.source, stop_event=self.stop_event):
                with self.lock:
                    self.latest_frame = state["frame"]
                    self.current_faces = state["faces"]
                    if state["new_registration"]:
                        self.events.append(state["new_registration"])
                logger.info(f"faces={len(state['faces'])} | new_reg={state['new_registration']}")
        except Exception as e:
            logger.error(f"producer crashed: {e}")

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Worker already running, ignoring start()")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("producer starts the thread.")

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        logger.info("producer stopped.")

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def get_current_faces(self):
        with self.lock:
            return self.current_faces

    def get_events(self):
        with self.lock:
            return list(self.events)