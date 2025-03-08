# Part 3: Vehicle Tracker
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
        """Initialize the VehicleTracker with camera parameters and conversion factor."""
        logger.info(f"Initializing VehicleTracker with pixels_per_meter={pixels_per_meter}")
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pixels_per_meter = pixels_per_meter
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def calculate_speed(self, vehicle_position_1: Tuple[float, float], 
                        vehicle_position_2: Tuple[float, float], time_difference: float) -> float:
        """Calculate vehicle speed based on position change and time difference."""
        if time_difference <= 0:
            logger.warning("Time difference is zero or negative, cannot calculate speed")
            return 0.0
        distance = np.linalg.norm(np.array(vehicle_position_2) - np.array(vehicle_position_1))
        speed = (distance / time_difference) * self.pixels_per_meter * 3.6  # Convert to km/h
        return speed

    def detect_vehicles(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detect vehicles in a frame using YOLO and return bounding boxes."""
        try:
            model = YOLO(os.getenv('YOLO_MODEL_PATH'))
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        try:
            results = model(frame)
            vehicle_bboxes = []
            for result in results.xywh[0]:
                x1, y1, w, h, conf = map(float, result[:5])
                if conf > 0.5:  # Confidence threshold
                    logger.debug(f"Detected vehicle with confidence {conf}")
                    vehicle_bboxes.append((int(x1), int(y1), int(w), int(h)))
            logger.info(f"Detected {len(vehicle_bboxes)} vehicles in frame")
            return vehicle_bboxes
        except Exception as e:
            logger.error(f"Error detecting vehicles: {e}")
            return []

    def extract_license_plate(self, vehicle_roi: np.ndarray) -> str:
        """Extract license plate text from a vehicle ROI using OCR."""
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
        """Process a video to detect and track vehicles, returning average speeds."""
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

            if time_difference is not None and time_difference <= 0:
                logger.warning(f"Non-positive time difference at frame {frame_count}")
                continue

            try:
                vehicle_bboxes = self.detect_vehicles(frame)
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue

            for x1, y1, w, h in vehicle_bboxes:
                vehicle_roi = frame[y1:y1 + h, x1:x1 + w]
                license_plate = self.extract_license_plate(vehicle_roi)
                if not license_plate:
                    logger.debug("Skipping vehicle with no license plate")
                    continue

                current_position = (x1 + w // 2, y1 + h // 2)

                if license_plate in prev_vehicle_positions and time_difference is not None:
                    prev_position = prev_vehicle_positions[license_plate]
                    speed = self.calculate_speed(prev_position, current_position, time_difference)
                    vehicle_speeds[license_plate].append(speed)
                    logger.info(f"Calculated speed for {license_plate}: {speed:.2f} km/h")
                else:
                    logger.debug(f"No previous position for {license_plate} yet")

                prev_vehicle_positions[license_plate] = current_position

        cap.release()
        logger.info("Finished processing video")

        vehicle_avg_speeds = []
        for license_plate, speeds in vehicle_speeds.items():
            if speeds:
                avg_speed = np.mean(speeds)
                vehicle_avg_speeds.append({'license_plate': license_plate, 'avg_speed': avg_speed})
                logger.info(f"Average speed for {license_plate}: {avg_speed:.2f} km/h")

        logger.info(f"Final results: {vehicle_avg_speeds}")
        return vehicle_avg_speeds