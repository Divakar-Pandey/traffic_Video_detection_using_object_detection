import torch
import logging
import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTracker:
    def __init__(self, max_age=10, min_iou=0.3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_iou = min_iou

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def update(self, detections: np.ndarray) -> Dict[int, Tuple[np.ndarray, int]]:
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return {k: (v['bbox'], v['class_id']) for k, v in self.tracks.items()}

        matched_track_indices = []
        matched_detection_indices = []

        if len(self.tracks) > 0:
            current_tracks = np.array([track['bbox'] for track in self.tracks.values()])
            current_track_ids = list(self.tracks.keys())

            iou_matrix = np.zeros((len(detections), len(current_tracks)))
            for i, det in enumerate(detections):
                for j, track in enumerate(current_tracks):
                    iou_matrix[i, j] = self.calculate_iou(det[:4], track)

            for i in range(len(detections)):
                j = np.argmax(iou_matrix[i])
                if iou_matrix[i, j] >= self.min_iou:
                    if j not in matched_track_indices:
                        matched_track_indices.append(j)
                        matched_detection_indices.append(i)

            # Update matched tracks
            for det_idx, track_idx in zip(matched_detection_indices, matched_track_indices):
                track_id = current_track_ids[track_idx]
                self.tracks[track_id].update({
                    'bbox': detections[det_idx, :4],
                    'confidence': detections[det_idx, 4],
                    'class_id': int(detections[det_idx, 5]),
                    'age': 0
                })

        # Initialize new tracks for unmatched detections
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]
        for i in unmatched_detections:
            self.tracks[self.next_id] = {
                'bbox': detections[i, :4],
                'confidence': detections[i, 4],
                'class_id': int(detections[i, 5]),
                'age': 0
            }
            self.next_id += 1

        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]

        return {k: (v['bbox'], v['class_id']) for k, v in self.tracks.items()}

class TrafficMonitor:
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ONNX runtime
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"Using device: {self.device}")
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_age=10, min_iou=0.3)
        
        # Class names for your model
        self.class_names = ['car', 'truck', 'bus', 'motorcycle']  # Update with your actual classes
        
        # Initialize frame counter
        self.frame_count = 0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img

    def detect_objects(self, frame: np.ndarray) -> np.ndarray:
        try:
            input_tensor = self.preprocess(frame)
            outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: input_tensor})
            
            # Print shape information for debugging
            logger.info(f"Model output shape: {outputs[0].shape}")
            
            detections = self.post_process(outputs[0])
            
            # Print detection information for debugging
            logger.info(f"Number of detections: {len(detections)}")
            if len(detections) > 0:
                logger.info(f"First detection: {detections[0]}")
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return np.array([])

    def post_process(self, output: np.ndarray) -> np.ndarray:
        predictions = output[0].transpose(1, 0)  # Transpose to (n_boxes, n_values)
        
        # Extract box coordinates and scores
        boxes = predictions[:, :4]  # First 4 values are box coordinates
        scores = predictions[:, 4:]  # Remaining values are class scores
        
        # Get class IDs and confidences
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > 0.3
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert boxes from centerx, centery, width, height to x1, y1, x2, y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)
        
        # Scale boxes to original image size
        boxes_xyxy[:, [0, 2]] *= 640
        boxes_xyxy[:, [1, 3]] *= 480
        
        # Combine detections
        detections = np.column_stack((boxes_xyxy, confidences, class_ids))
        return detections.astype(np.float32)

    def process_video(self, video_path: str, output_path: Optional[str] = None, width: int = 640, height: int = 480):
        logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"File does not exist at path: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (width, height))
                detections = self.detect_objects(frame_resized)
                tracked_objects = self.tracker.update(detections)
                
                # Draw bounding boxes and IDs
                for track_id, (bbox, class_id) in tracked_objects.items():
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    # Draw bbox
                    cv2.rectangle(frame_resized, 
                                (x1, y1), 
                                (x2, y2),
                                (0, 255, 0), 2)
                    
                    # Draw ID and class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                    label = f"ID: {track_id}, {class_name}"
                    cv2.putText(frame_resized,
                              label,
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 255, 0),
                              2)
                
                cv2.imshow("Traffic Monitor", frame_resized)
                if writer:
                    writer.write(frame_resized)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                self.frame_count += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

def main():
    monitor = TrafficMonitor(
        model_path=r"C:\traffic video analysis\ultralytics\model folder\runs\detect\yolov8custom_model_traffic\weights\best.onnx",
        output_dir=r"C:\traffic video analysis\output video"
    )

    monitor.process_video(
        video_path=r"C:\traffic video analysis\input video\12535442_3840_2160_30fps.mp4",
        output_path=r"C:\traffic video analysis\output video\output_video.mp4"
    )

if __name__ == "__main__":
    main()