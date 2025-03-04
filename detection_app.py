import streamlit as st
import cv2
import cvzone
import tempfile
import os
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO  

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()  # Load YOLOv8
        self.class_names = self.model.names  # Get class names from YOLOv8
        self.tracker = self.initialize_tracker()
        self.class_counts = defaultdict(int)  # Store counts per object class
        self.crossed_ids = set()  # Track objects that have crossed the line
        self.line_position = 250  # Default position of the line (Y-coordinate for horizontal or X for vertical)
        self.line_direction = "horizontal"  # Default line direction
        self.line_speed = 5  # Line speed for moving back and forth

    def load_model(self):
        """Load YOLOv8 pre-trained model."""
        st.write("Loading YOLOv8 model...")
        model_path = r"C:\traffic video analysis\ultralytics\model folder\runs\detect\yolov8custom_model_traffic\weights\best.onnx"
        model = YOLO(model_path)  
        st.write("YOLOv8 model loaded successfully")
        return model

    def initialize_tracker(self):
        """Initialize DeepSORT Tracker."""
        st.write("Initializing DeepSort tracker...")
        tracker = DeepSort(max_age=5, n_init=2, max_cosine_distance=0.3, nn_budget=100)
        st.write("DeepSort tracker initialized successfully")
        return tracker

    def predict(self, img):
        """Run YOLOv8 inference."""
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        """Extract bounding boxes and return detections."""
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                conf = round(box.conf[0].item(), 2)

                # Confidence threshold
                if conf > 0.65:
                    detections.append(((x1, y1, w, h), conf, class_name))

        return detections, img

    def track_detect(self, detections, img):
        """Track objects and count those that cross the line."""
        # Draw the line depending on its direction
        if self.line_direction == "horizontal":
            cv2.line(img, (50, self.line_position), (600, self.line_position), (255, 0, 0), 2)  # Horizontal line
        elif self.line_direction == "vertical":
            cv2.line(img, (self.line_position, 50), (self.line_position, 400), (255, 0, 0), 2)  # Vertical line

        tracks = self.tracker.update_tracks(detections, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue

            class_name = track.det_class
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of bounding box

            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)  # Mark center with yellow color

            # Check if object crosses the line based on direction
            if self.line_direction == "horizontal":
                if cy > self.line_position and track.track_id not in self.crossed_ids:
                    self.crossed_ids.add(track.track_id)
                    self.class_counts[class_name] += 1  # Increment count
            elif self.line_direction == "vertical":
                if cx > self.line_position and track.track_id not in self.crossed_ids:
                    self.crossed_ids.add(track.track_id)
                    self.class_counts[class_name] += 1  # Increment count

            # bounding box with thinner edges and deep blue color
            cvzone.putTextRect(img, f"{class_name} {track.track_id}", (x1, y1), scale=0.8, thickness=1, colorR=(255, 255, 0))
            cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2, colorR=(0, 0, 255))  # Thinner blue box

        # Display count info with cleaner font
        y_offset = 30
        for class_name, count in self.class_counts.items():
            cv2.putText(img, f"{class_name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        return img

    def process_video(self, video_file):
        """Process video frame-by-frame."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            temp_video_path = tmp_file.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Error: Unable to open video file.")
            return

        frame_skip = 2  # Process every 2nd frame
        frame_count = 0

        video_frame_placeholder = st.empty()  # Create a Streamlit placeholder

        while True:
            success, img = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            img = cv2.resize(img, (640, 360))

            results = self.predict(img)
            detections, img = self.plot_boxes(results, img)
            tracked_img = self.track_detect(detections, img)

            video_frame_placeholder.image(tracked_img, channels="BGR", use_container_width=True)

        cap.release()
        os.remove(temp_video_path)  # Delete temporary file

    def __call__(self):
        st.title("Traffic Object Detection & Counting")
        st.write("Upload a video for YOLOv8-based detection & counting.")

        # Configuration for the line (direction and speed)
        self.line_direction = st.selectbox("Choose line direction", ["horizontal", "vertical"], index=0)
        self.line_position = st.slider("Set the position of the counting line:", 50, 600, 250)
        self.line_speed = st.slider("Set line speed (number of pixels per frame):", 1, 10, 5)

        # Toggle line movement
        move_line = st.checkbox("Enable line movement (back and forth)")

        if move_line:
            self.line_position += self.line_speed
            if self.line_position >= 600 or self.line_position <= 50:
                self.line_speed = -self.line_speed  

        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if video_file:
            st.write("Processing Video...")
            self.process_video(video_file)
        else:
            st.warning("Please upload a video file.")

if __name__ == "__main__":
    detector = ObjectDetection()
    detector()  