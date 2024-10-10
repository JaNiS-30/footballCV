import supervision as sv
from roboflow import get_model
import numpy as np

class PitchKeypointsDetect:
    def __init__(self, api_key, model_id="football-field-detection-f07vi/14", confidence_threshold=0.3):
        self.api_key = api_key
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.model = get_model(model_id=self.model_id, api_key=self.api_key)
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex('#FF1493'), radius=8)
    
    def detect_keypoints(self, frame):
        result = self.model.infer(frame, confidence=self.confidence_threshold)[0]
        key_points = sv.KeyPoints.from_inference(result)

        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        frame_reference_key_points = sv.KeyPoints(
            xy=frame_reference_points[np.newaxis, ...])

        return frame_reference_key_points

    def annotate_frame(self, frame, key_points):
        annotated_frame = frame.copy()
        annotated_frame = self.vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=key_points
        )
        return annotated_frame