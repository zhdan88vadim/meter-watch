import cv2

from app.person_tracker import PersonTracker
from app.video_buffer import VideoBuffer
from app.rate_limiter import SimpleRateLimiter
from app.infra.yolo_detector import YoloDetector
from tests.fakes import FakeRepository, FakeStore


VIDEO_PATH = "./tests/assets/track_1.mp4"
VIDEO_PATH_2 = "./tests/assets/track_3.mp4"
VIDEO_PATH_3 = "./tests/assets/track_2.mp4"
VIDEO_PATH_WITHOUT_ACTIVITY = "./tests/assets/without_person.mp4"

class FakeKeys:
    human_last_seen = "last_seen"
    human_last_seen_str = "last_seen_str"

def read_frames(path, max_frames=None):
    """Yield frames from a video file."""
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Cannot open {path}"
    try:
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            count += 1
            if max_frames and count >= max_frames:
                break
    finally:
        cap.release()


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


def make_tracker(clock=None):
    clock = clock or FakeClock(0.0)

    detector = YoloDetector("yolov8n.pt")
    store = FakeStore()
    repo = FakeRepository()

    buffer = VideoBuffer(buffer_seconds=1, fps=10, clock=clock)    
    rate_limiter = SimpleRateLimiter(min_interval_seconds=0, clock=clock)

    tracker = PersonTracker(
        keys=FakeKeys(),  
        detector=detector,
        store=store,
        db_repository=repo,
        buffer=buffer,
        rate_limiter=rate_limiter,
        post_roll_seconds=0,           # <-- avoid real sleep in tests!
        frame_skip=1,
        clock=clock,
        source=VIDEO_PATH,
    )
    return tracker, store, repo, clock



def test_detection_logged_with_real_video():    
    tracker, store, repo, clock = make_tracker()

    for i, frame in enumerate(read_frames(VIDEO_PATH, max_frames=100)):
        tracker.process_frame(frame)
        clock.advance(0.1)  # 100ms per frame

    assert store.get("last_seen") is not None
    assert repo.detected



def test_detection_logged_with_real_video_file_2():    
    tracker, store, repo, clock = make_tracker()

    for i, frame in enumerate(read_frames(VIDEO_PATH_2, max_frames=100)):
        tracker.process_frame(frame)
        clock.advance(0.1)  # 100ms per frame

    assert store.get("last_seen") is not None
    assert repo.detected


def test_detection_logged_with_real_video_file_3():    
    tracker, store, repo, clock = make_tracker()

    for i, frame in enumerate(read_frames(VIDEO_PATH_3, max_frames=100)):
        tracker.process_frame(frame)
        clock.advance(0.1)  # 100ms per frame

    assert store.get("last_seen") is not None
    assert repo.detected


def test_no_detections_does_not_log():
    tracker, store, repo, clock = make_tracker()

    for frame in read_frames(VIDEO_PATH_WITHOUT_ACTIVITY, max_frames=100):
        tracker.process_frame(frame)
        clock.advance(0.1)

    assert not repo.detected
    assert store.get("last_seen") is None