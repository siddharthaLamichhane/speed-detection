# import cv2
# import numpy as np
# import re
# from datetime import datetime
# import easyocr  # Replace PaddleOCR with EasyOCR
# from django.conf import settings
# import os
# from typing import List, Optional, Tuple
# import logging
# from ultralytics import YOLO

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class LicensePlateReader:
#     def __init__(self):
#         """Initialize the license plate reader with YOLOv8 model and OCR."""
#         self.model = None
#         self.ocr = None
#         self.className = ["License"]
#         self._initialize_models()
        
#     def _initialize_models(self):
#         """Initialize YOLOv8 and OCR models with error handling."""
#         try:
#             # Initialize YOLOv8 model
#             if not os.path.exists(settings.YOLO_MODEL_PATH):
#                 raise FileNotFoundError(f"YOLO model not found at {settings.YOLO_MODEL_PATH}")
            
#             # Load YOLOv8 model
#             self.model = YOLO(settings.YOLO_MODEL_PATH)
            
#             # Initialize EasyOCR
#             self.ocr = easyocr.Reader(['en'])  # Initialize for English
            
#             logger.info("Models initialized successfully")
#         except Exception as e:
#             logger.error(f"Error initializing models: {str(e)}")
#             raise
            
#     def _validate_video(self, video_path: str) -> bool:
#         """Validate video file exists and is readable."""
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video file not found: {video_path}")
            
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video file: {video_path}")
            
#         # Check if video has frames
#         ret, frame = cap.read()
#         cap.release()
#         return ret
    
#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """Preprocess frame for better detection."""
#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Apply adaptive thresholding
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY, 11, 2)
        
#         # Denoise
#         denoised = cv2.fastNlMeansDenoising(thresh)
        
#         return denoised
    
#     def _extract_text(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
#         """Extract text from a region using OCR."""
#         x1, y1, x2, y2 = bbox
#         roi = frame[y1:y2, x1:x2]
        
#         # Ensure ROI is not empty
#         if roi.size == 0:
#             return ""
            
#         # OCR processing with EasyOCR
#         results = self.ocr.readtext(roi)
#         text = ""
        
#         for (bbox, text_result, prob) in results:
#             if prob > 0.6:  # 60% confidence threshold
#                 text = text_result
#                 # Clean up the text
#                 text = re.sub('[\W]', '', text)
#                 text = text.replace("O", "0")
#                 break
                
#         return text
    
#     def extract_license_plate(self, video_path: str) -> List[str]:
#         """
#         Extract all unique license plates detected in the video using YOLOv8.
#         Args:
#             video_path: Path to the video file
#         Returns:
#             List of unique detected license plates
#         """
#         try:
#             # Validate video
#             self._validate_video(video_path)
            
#             cap = cv2.VideoCapture(video_path)
#             license_plates = set()
            
#             frame_count = 0
#             frame_skip = 10  # Process every 10th frame
            
#             while True:
#                 ret, frame = cap.read()
#                 if not ret or frame is None or frame.size == 0:
#                     logger.error("Empty frame encountered or failed to read frame.")
#                     break
                
#                 frame_count += 1
#                 if frame_count % frame_skip != 0:
#                     continue
                
#                 # Preprocess frame
#                 processed_frame = self._preprocess_frame(frame)
#                 processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)  # Convert to RGB
                
#                 # Run YOLOv8 detection
#                 results = self.model(processed_frame_rgb, conf=0.3)
                
#                 # Process YOLOv8 results
#                 for result in results:
#                     boxes = result.boxes
#                     logger.info(f"Detected boxes: {boxes}")  # Debugging output
                    
#                     for box in boxes:
#                         # Get coordinates
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
#                         # Extract text from detected region
#                         text = self._extract_text(processed_frame, (x1, y1, x2, y2))
                        
#                         if text:
#                             license_plates.add(text)
                            
#                             # Optional: Log detection results
#                             logger.info(f"Detected license plate: {text}")
                
#             cap.release()
            
#             return list(license_plates)
            
#         except Exception as e:
#             logger.error(f"Error processing video: {str(e)}")
#             raise

# # Global instance for caching
# _license_plate_reader = None

# def get_license_plate_reader() -> LicensePlateReader:
#     """Get or create the license plate reader instance."""
#     global _license_plate_reader
#     if _license_plate_reader is None:
#         _license_plate_reader = LicensePlateReader()
#     return _license_plate_reader

# def extract_license_plate(video_path: str) -> List[str]:
#     """Wrapper function to extract license plates from video."""
#     reader = get_license_plate_reader()
#     return reader.extract_license_plate(video_path)



#scsdcsdcsdcsdcsdcsdddfbfgbghghn
# import easyocr
# import cv2
# import numpy as np
# import re
# import os
# from django.conf import settings
# from typing import List, Optional, Tuple
# import logging
# from .video_processor import SpeedCalculator  # Import SpeedCalculator class

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class LicensePlateReader:
#     def __init__(self, speed_calculator: Optional[SpeedCalculator] = None):
#         """Initialize the license plate reader with YOLOv8 model and OCR."""
#         self.model = None
#         self.ocr = None
#         self.className = ["License"]
#         self.speed_calculator = speed_calculator or SpeedCalculator()  # Initialize SpeedCalculator
#         self._initialize_models()

#     def _initialize_models(self):
#         """Initialize YOLOv8 and OCR models with error handling."""
#         try:
#             # Initialize YOLOv8 model
#             if not os.path.exists(settings.YOLO_MODEL_PATH):
#                 raise FileNotFoundError(f"YOLO model not found at {settings.YOLO_MODEL_PATH}")
            
#             # Load YOLOv8 model
#             from ultralytics import YOLO
#             self.model = YOLO(settings.YOLO_MODEL_PATH)
            
#             # Initialize EasyOCR
#             self.ocr = easyocr.Reader(['en'])  # Initialize for English
            
#             logger.info("Models initialized successfully")
#         except Exception as e:
#             logger.error(f"Error initializing models: {str(e)}")
#             raise

#     def _validate_video(self, video_path: str) -> bool:
#         """Validate video file exists and is readable."""
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video file not found: {video_path}")
            
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video file: {video_path}")
            
#         # Check if video has frames
#         ret, frame = cap.read()
#         cap.release()
#         return ret

#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """Preprocess frame for better detection."""
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY, 11, 2)
#         denoised = cv2.fastNlMeansDenoising(thresh)
#         return denoised

#     def _extract_text(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
#         """Extract text from a region using OCR."""
#         x1, y1, x2, y2 = bbox
#         roi = frame[y1:y2, x1:x2]
#         if roi.size == 0:
#             return ""
            
#         results = self.ocr.readtext(roi)
#         text = ""
        
#         for (bbox, text_result, prob) in results:
#             if prob > 0.6:  # 60% confidence threshold
#                 text = text_result
#                 text = re.sub('[\W]', '', text)  # Clean text
#                 text = text.replace("O", "0")  # Fix 'O' to '0'
#                 break
                
#         return text

#     def extract_license_plate_and_speed(self, video_path: str) -> List[Tuple[str, float]]:
#         """Extract license plates and their associated speed from the video."""
#         try:
#             # Validate video
#             self._validate_video(video_path)
            
#             cap = cv2.VideoCapture(video_path)
#             frame_count = 0
#             frame_skip = 10  # Process every 10th frame
#             license_plates_and_speeds = []

#             prev_frame = None
#             prev_kp = None
#             prev_des = None

#             while True:
#                 ret, frame = cap.read()
#                 if not ret or frame is None or frame.size == 0:
#                     break
                
#                 frame_count += 1
#                 if frame_count % frame_skip != 0:
#                     continue

#                 processed_frame = self._preprocess_frame(frame)
#                 processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)  # Convert to RGB
                
#                 try:
#                     # Run YOLOv8 detection
#                     results = self.model(processed_frame_rgb, conf=0.3)
#                     for result in results:
#                         boxes = result.boxes
#                         for box in boxes:
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])
#                             license_plate = self._extract_text(processed_frame, (x1, y1, x2, y2))
                            
#                             if license_plate:
#                                 # Calculate speed for the detected vehicle
#                                 speed = self.speed_calculator.calculate_speed(video_path)[0]
#                                 license_plates_and_speeds.append((license_plate, speed))
#                                 logger.info(f"License Plate: {license_plate}, Speed: {speed} km/h")
#                 except Exception as e:
#                     logger.warning(f"Error running YOLOv8 detection: {str(e)}")
#                     continue

#             cap.release()
#             return license_plates_and_speeds
#         except FileNotFoundError as e:
#             logger.error(f"File not found: {str(e)}")
#             raise
#         except ValueError as e:
#             logger.error(f"Invalid value: {str(e)}")
#             raise
#         except Exception as e:
#             logger.error(f"Error extracting license plates and speed: {str(e)}") 
 
        
import easyocr
import cv2
import numpy as np
import re
import os
from django.conf import settings
from typing import List, Optional, Tuple
import logging
import easyocr
from typing import List, Tuple
from .video_processor import VehicleTracker

class LicensePlateReader:
    def __init__(self):
        self.ocr = easyocr.Reader(['en'])
        print("self.ocr",self.ocr)
        self.speed_calculator = VehicleTracker()
        print("vspeed_calculator",self.speed_calculator)
        
        

    def extract_license_plate_and_speed(self, video_path: str) -> List[Tuple[str, float]]:
        # Get the license plates and speeds from the video
        license_plate_and_speeds = []

        # Calculate speed and license plate recognition
        vehicle_avg_speeds = self.speed_calculator.detect_and_track_vehicles(video_path)

        # Recognize the license plates for the detected vehicles
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for result in vehicle_avg_speeds:
                license_plate = None
                ocr_result = self.ocr.readtext(frame)
                for detection in ocr_result:
                    if detection[1] == result[0]:  # Match license plate with speed
                        license_plate = detection[1]
                        break

                if license_plate:
                    license_plate_and_speeds.append((license_plate, result[1]))

        cap.release()

        return license_plate_and_speeds
