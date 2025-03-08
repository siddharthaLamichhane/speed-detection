import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import defaultdict
from ultralytics import YOLO
import pytesseract
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class VehicleTracker:
    def __init__(self, camera_matrix: Optional[np.ndarray] = None, 
                 dist_coeffs: Optional[np.ndarray] = None,
                 pixels_per_meter: float = 100):
        logger.info(f"Initializing VehicleTracker with pixels_per_meter={pixels_per_meter}")
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pixels_per_meter = pixels_per_meter
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        try:
            self.model = YOLO(os.getenv('YOLO_MODEL_PATH'))
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_vehicles(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detect vehicles (or license plates) in a frame using YOLO and return bounding boxes."""
        try:
            results = self.model(frame)  # Run YOLO inference
            logger.info(f"YOLO raw output type: {type(results)}")
            vehicle_bboxes = []

            # Handle list of Results objects
            if isinstance(results, list):
                for result in results:
                    if hasattr(result, 'boxes'):  # Check for boxes attribute
                        for box in result.boxes:  # Iterate over detected boxes
                            # Extract coordinates (xywh format)
                            x, y, w, h = map(int, box.xywh[0])  # xywh returns a tensor, [0] for first item
                            conf = float(box.conf[0])  # Confidence score
                            if conf > 0.5:  # Confidence threshold
                                logger.debug(f"Detected object at x={x}, y={y}, w={w}, h={h} with confidence {conf}")
                                vehicle_bboxes.append((x, y, w, h))
                    else:
                        logger.warning(f"Result object lacks 'boxes' attribute: {result}")
            else:
                logger.error(f"Unsupported YOLO output format: {type(results)}")
                return []

            logger.info(f"Detected {len(vehicle_bboxes)} vehicles in frame")
            return vehicle_bboxes
        except Exception as e:
            logger.error(f"Error detecting vehicles: {e}")
            return []
        
    def calculate_speed(self, vehicle_position_1: Tuple[float, float], 
                        vehicle_position_2: Tuple[float, float], time_difference: float) -> float:
        if time_difference <= 0:
            logger.warning("Time difference is zero or negative, cannot calculate speed")
            return 0.0
        distance = np.linalg.norm(np.array(vehicle_position_2) - np.array(vehicle_position_1))
        speed = (distance / time_difference) * self.pixels_per_meter * 3.6  # km/h
        return speed

    def extract_license_plate(self, vehicle_roi: np.ndarray) -> str:
        try:
            plate_text = pytesseract.image_to_string(vehicle_roi, config='--psm 8').strip()
            if not plate_text:
                logger.warning("No license plate text extracted from ROI")
            else:
                logger.info(f"Extracted license plate: {plate_text}")
            return plate_text
        except Exception as e:
            logger.error(f"Error extracting license plate: {e}")
            return ""

    def detect_and_track_vehicles(self, video_path: str) -> List[dict]:
        logger.info(f"Starting video processing for {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file at {video_path}")
            raise ValueError(f"Could not open video file at {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video properties: FPS={fps}, Width={frame_width}, Height={frame_height}")

        vehicle_speeds = defaultdict(list)
        prev_vehicle_positions = {}
        prev_timestamp = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Reached end of video or failed to read frame")
                break
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_difference = (timestamp - prev_timestamp) / 1000 if prev_timestamp is not None else None
            prev_timestamp = timestamp

            vehicle_bboxes = self.detect_vehicles(frame)
            for x, y, w, h in vehicle_bboxes:
                vehicle_roi = frame[y:y+h, x:x+w]
                license_plate = self.extract_license_plate(vehicle_roi)
                if not license_plate:
                    continue

                current_position = (x + w/2, y + h/2)
                if license_plate in prev_vehicle_positions and time_difference is not None:
                    prev_position = prev_vehicle_positions[license_plate]
                    speed = self.calculate_speed(prev_position, current_position, time_difference)
                    vehicle_speeds[license_plate].append(speed)
                    logger.info(f"Calculated speed for {license_plate}: {speed:.2f} km/h")
                prev_vehicle_positions[license_plate] = current_position

        cap.release()
        logger.info("Finished processing video")

        vehicle_avg_speeds = [
            {'license_plate': lp, 'avg_speed': np.mean(speeds)}
            for lp, speeds in vehicle_speeds.items() if speeds
        ]
        logger.info(f"Final results: {vehicle_avg_speeds}")
        return vehicle_avg_speeds