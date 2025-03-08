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
        # creates an ocr object for english text
        self.ocr = easyocr.Reader(['en'])
        """
        easyocr.Reader(['en']) creates an object that can read English text ('en' specifies the language).
        This object has methods like readtext(), which analyzes an image and returns detected text.
        """

        # Instantiates a VehicleTracker object ( to handle speed detection).
        self.speed_calculator = VehicleTracker()
        # print("vspeed_calculator",self.speed_calculator)
        
        
    # input parameter: video, returns: tuple: (lisence plate string, speed)
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
