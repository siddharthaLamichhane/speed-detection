import cv2
from vehicle_processing.video_processor import VehicleTracker  # Confirm this matches your structure exceeding rate limit
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test the VehicleTracker
video_path = "D:/LEARNING/Django/speed-detection/media/videos/file_sample_G5FuSOk.mp4"
tracker = VehicleTracker()

# Process the video
try:
    results = tracker.detect_and_track_vehicles(video_path)
    logger.info(f"Results: {results}")
except Exception as e:
    logger.error(f"Error in test: {e}")

# Test a single frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if ret:
    bboxes = tracker.detect_vehicles(frame)
    logger.info(f"Detected bounding boxes: {bboxes}")
else:
    logger.error("Failed to read video frame")
cap.release()