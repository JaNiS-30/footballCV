from ultralytics import YOLO
import supervision as sv
import pickle
import os
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()


class TrackerPitch:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.3)
            detections += detections_batch
        return detections

    def get_pitch_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {"keypoints": []}
        detections = self.detect_frames(frames)

        for frame_num, (frame, detection) in enumerate(zip(frames, detections)):
            if hasattr(detection, "keypoints") and detection.keypoints is not None:
                key_points = detection.keypoints
                frame_reference_points = key_points.xy[0].numpy()
                confidences = key_points.conf[0].numpy()

                tracks["keypoints"].append(
                    {
                        "frame_num": frame_num,
                        "confidences": confidences,
                        "filtered_points": frame_reference_points,
                    }
                )

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
