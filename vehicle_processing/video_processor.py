
# import cv2
# import numpy as np
# from typing import Tuple, Optional

# class SpeedCalculator:
#     def __init__(self, camera_matrix: Optional[np.ndarray] = None, 
#                  dist_coeffs: Optional[np.ndarray] = None,
#                  pixels_per_meter: float = 100):
#         """
#         Initialize speed calculator with camera calibration parameters
#         Args:
#             camera_matrix: Camera calibration matrix
#             dist_coeffs: Distortion coefficients
#             pixels_per_meter: Number of pixels per meter in the video
#         """
#         self.camera_matrix = camera_matrix
#         self.dist_coeffs = dist_coeffs
#         self.pixels_per_meter = pixels_per_meter
#         self.feature_detector = cv2.SIFT_create()
#         self.matcher = cv2.BFMatcher()

#     def calculate_speed(self, video_path: str) -> Tuple[float, list]:
#         """
#         Calculate vehicle speed from video
#         Args:
#             video_path: Path to the video file
#         Returns:
#             Tuple of (average_speed, speed_history)
#         """
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("Could not open video file")

#         # Get video properties
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # Initialize variables
#         prev_frame = None
#         prev_kp = None
#         prev_des = None
#         speeds = []
#         frame_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             # Convert to grayscale
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect features
#             kp, des = self.feature_detector.detectAndCompute(gray, None)

#             # 🚀 **Fix 1: Ensure descriptors exist before matching**
#             if des is None or prev_des is None or len(des) < 2 or len(prev_des) < 2:
#                 prev_frame = frame
#                 prev_kp = kp
#                 prev_des = des
#                 continue  # Skip this frame
            
#             # Match features between frames
#             matches = self.matcher.knnMatch(des, prev_des, k=2)

#             # Apply ratio test to get good matches
#             good_matches = []
#             for m, n in matches:
#                 if m.distance < 0.75 * n.distance:
#                     good_matches.append(m)

#             # 🚀 **Fix 2: Ensure enough good matches before proceeding**
#             if len(good_matches) < 10:
#                 prev_frame = frame
#                 prev_kp = kp
#                 prev_des = des
#                 continue  # Skip this frame
            
#             # Extract matched points
#             src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
#             dst_pts = np.float32([prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

#             # 🚀 **Fix 3: Ensure shapes match before subtraction**
#             if src_pts.shape != dst_pts.shape:
#                 print(f"Shape mismatch! src_pts: {src_pts.shape}, dst_pts: {dst_pts.shape}")
#                 prev_frame = frame
#                 prev_kp = kp
#                 prev_des = des
#                 continue  # Skip this frame

#             # Calculate average movement in pixels
#             movement = np.mean(np.linalg.norm(dst_pts - src_pts, axis=1))  # 🚀 **Fixed axis from 2 to 1**

#             # Convert to speed (km/h)
#             speed = (movement * self.pixels_per_meter * fps * 3.6) / 1000
#             speeds.append(speed)

#             # Update previous frame data
#             prev_frame = frame
#             prev_kp = kp
#             prev_des = des
#             frame_count += 1

#         cap.release()

#         if not speeds:
#             return 0.0, []

#         return np.mean(speeds), speeds

# def calculate_speed(video_path: str) -> float:
#     """
#     Wrapper function to calculate speed from video
#     Args:
#         video_path: Path to the video file
#     Returns:
#         Average speed in km/h
#     """
#     calculator = SpeedCalculator()
#     avg_speed, _ = calculator.calculate_speed(video_path)
#     return avg_speed

# import cv2
# import numpy as np
# import logging
# from typing import List, Tuple, Optional
# from ultralytics import YOLO
# from datetime import datetime
# import easyocr  # For license plate recognition
# from collections import defaultdict

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class VehicleTracker:
#     def __init__(self, camera_matrix: Optional[np.ndarray] = None, 
#                  dist_coeffs: Optional[np.ndarray] = None,
#                  pixels_per_meter: float = 100):
#         """
#         Initialize vehicle tracker with camera calibration parameters
#         Args:
#             camera_matrix: Camera calibration matrix
#             dist_coeffs: Distortion coefficients
#             pixels_per_meter: Number of pixels per meter in the video
#         """
#         self.camera_matrix = camera_matrix
#         self.dist_coeffs = dist_coeffs
#         self.pixels_per_meter = pixels_per_meter
#         self.feature_detector = cv2.SIFT_create()
#         self.matcher = cv2.BFMatcher()
#         self.ocr = easyocr.Reader(['en'])  # Initialize EasyOCR for license plate recognition

#     def detect_and_track_vehicles(self, video_path: str) -> List[Tuple[str, float]]:
#         """
#         Detect and track multiple vehicles in the video, calculate their speed and recognize their license plates.
#         Args:
#             video_path: Path to the video file
#         Returns:
#             List of tuples containing vehicle license plate and speed in km/h
#         """
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video file at {video_path}")

#         # Get video properties
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         vehicle_speeds = defaultdict(list)  # Dictionary to store speeds for each vehicle
#         vehicle_plate_map = {}  # Dictionary to store license plates for each vehicle
#         frame_count = 0

#         # Initialize vehicle detection model (YOLO for vehicle detection)
#         model = YOLO("yolov8.pt")  # Load a pre-trained YOLO model

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Detect vehicles in the frame
#             results = model(frame)  # Detect vehicles using YOLO
#             for result in results.xywh[0]:
#                 if result[4] > 0.5:  # Confidence threshold
#                     x1, y1, w, h = map(int, result[:4])
#                     vehicle_roi = frame[y1:y1 + h, x1:x1 + w]

#                     # Recognize license plate
#                     ocr_result = self.ocr.readtext(vehicle_roi)
#                     license_plate = None
#                     if ocr_result:
#                         license_plate = ocr_result[0][-2]  # Get the recognized license plate

#                     # Track the vehicle's movement
#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     kp, des = self.feature_detector.detectAndCompute(gray, None)
#                     if des is not None:
#                         matches = self.matcher.knnMatch(des, des, k=2)
#                         good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

#                         if len(good_matches) > 10:
#                             movement = np.mean([np.linalg.norm(kp[m.queryIdx].pt - kp[m.trainIdx].pt) for m in good_matches])
#                             speed = (movement * self.pixels_per_meter * fps * 3.6) / 1000  # Convert to km/h

#                             # Add speed to the vehicle's record
#                             if license_plate:
#                                 vehicle_speeds[license_plate].append(speed)
#                                 if license_plate not in vehicle_plate_map:
#                                     vehicle_plate_map[license_plate] = license_plate

#             frame_count += 1

#         cap.release()

#         # Calculate average speed for each vehicle
#         vehicle_avg_speeds = []
#         for license_plate, speeds in vehicle_speeds.items():
#             if speeds:
#                 avg_speed = np.mean(speeds)
#                 vehicle_avg_speeds.append((license_plate, avg_speed))

#         return vehicle_avg_speeds



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
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pixels_per_meter = pixels_per_meter
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def calculate_speed(self, vehicle_position_1: Tuple[float, float], 
                        vehicle_position_2: Tuple[float, float], time_difference: float) -> float:
        distance = np.linalg.norm(np.array(vehicle_position_2) - np.array(vehicle_position_1))
        speed = (distance / time_difference) * self.pixels_per_meter * 3.6  # Convert to km/h
        return speed

    def detect_vehicles(self, frame) -> List[Tuple[int, int, int, int]]:
        model = YOLO(os.getenv('YOLO_MODEL_PATH'))  # Load the YOLO model
        results = model(frame)  # Perform detection
        vehicle_bboxes = []
        for result in results.xywh[0]:
            if result[4] > 0.5:  # Confidence threshold
                x1, y1, w, h = map(int, result[:4])
                vehicle_bboxes.append((x1, y1, w, h))
        return vehicle_bboxes

    def extract_license_plate(self, vehicle_roi: np.ndarray) -> str:
        plate_text = pytesseract.image_to_string(vehicle_roi, config='--psm 8')
        return plate_text.strip()

    def detect_and_track_vehicles(self, video_path: str) -> List[dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vehicle_speeds = defaultdict(list)
        vehicle_plate_map = {}
        prev_vehicle_positions = {}

        prev_timestamp = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get timestamp in milliseconds
            if prev_timestamp is not None:
                time_difference = (timestamp - prev_timestamp) / 1000  # Convert to seconds
            prev_timestamp = timestamp

            vehicle_bboxes = self.detect_vehicles(frame)

            for x1, y1, w, h in vehicle_bboxes:
                vehicle_roi = frame[y1:y1 + h, x1:x1 + w]
                license_plate = self.extract_license_plate(vehicle_roi)  # Get license plate from ROI
                current_position = (x1 + w // 2, y1 + h // 2)  # Center position of the vehicle

                # Track vehicle and compute speed
                if license_plate in prev_vehicle_positions:
                    prev_position = prev_vehicle_positions[license_plate]
                    speed = self.calculate_speed(prev_position, current_position, time_difference)

                    # Store speed information
                    vehicle_speeds[license_plate].append(speed)

                # Update the previous position for the vehicle
                prev_vehicle_positions[license_plate] = current_position

        cap.release()

        # Calculate average speed for each vehicle
        vehicle_avg_speeds = []
        for license_plate, speeds in vehicle_speeds.items():
            if speeds:
                avg_speed = np.mean(speeds)
                vehicle_avg_speeds.append({
                    'license_plate': license_plate,
                    'avg_speed': avg_speed
                })

        return vehicle_avg_speeds