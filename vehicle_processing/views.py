


# Part 1: Django View for Video Upload and Processing
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from .video_processor import VehicleTracker
from .license_plate_reader import LicensePlateReader
from .models import User, UserProfile, Violation
import json
import os
import logging
import magic as magic_win
from django.utils import timezone

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/avi', 'video/mpeg']
MAX_VIDEO_SIZE = settings.MAX_VIDEO_SIZE
SPEED_LIMIT = settings.SPEED_LIMIT

def send_violation_email(user: User, speed: float, timestamp: timezone.datetime) -> None:
    """
    Sends an email to the user notifying them of a speed violation.
    """
    try:
        subject = f'Speed Violation Alert - License Plate: {user.profile.license_plate}'
        context = {
            'username': user.username,
            'license_plate': user.profile.license_plate,
            'speed': speed,
            'speed_limit': SPEED_LIMIT,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'excess_speed': speed - SPEED_LIMIT
        }
        html_message = render_to_string('vehicle_processing/email/violation_notification.html', context)
        plain_message = render_to_string('vehicle_processing/email/violation_notification.txt', context)
        send_mail(
            subject=subject,
            message=plain_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
        logger.info(f"Violation email sent to {user.email}")
    except Exception as e:
        logger.error(f"Failed to send violation email: {str(e)}")

def validate_video_file(file) -> None:
    """
    Validates the uploaded video file for size and type.
    """
    if file.size > MAX_VIDEO_SIZE:
        raise ValidationError(f"Video file too large. Max size is {MAX_VIDEO_SIZE/1024/1024}MB")
    file_type = magic_win.from_buffer(file.read(1024), mime=True)
    file.seek(0)  # Reset file pointer after reading
    if file_type not in ALLOWED_VIDEO_TYPES:
        raise ValidationError(f"Invalid file type. Allowed types: {', '.join(ALLOWED_VIDEO_TYPES)}")

@csrf_exempt
def upload_video(request) -> JsonResponse:
    """
    Handles video upload, processes it to detect license plates and speeds,
    and records violations if the speed exceeds the limit.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file uploaded'}, status=400)
    
    video = request.FILES['video']
    
    try:
        # Validate the uploaded video file
        validate_video_file(video)
        print("Sucessful validation of video File")

        # Save the video file to the specified location
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'videos'))
        filename = fs.save(video.name, video)
        video_path = fs.path(filename)
        print(f"Video {filename} is saved at {video_path}")

        # Extract license plates and speeds from the video
        license_plate_reader = LicensePlateReader()
        print("Liscense plate reader output:",license_plate_reader)
        license_plate_and_speeds = license_plate_reader.extract_license_plate_and_speed(video_path)
        print("extracted liscence plate and speed [tuple]: ",license_plate_and_speeds)
        if not license_plate_and_speeds:
            return JsonResponse({'error': 'No license plates detected in the video', 'file_url': fs.url(filename)}, status=400)

        # Process each detected license plate and speed
        violations = []
        for license_plate, speed in license_plate_and_speeds:
            try:
                print("inside")
                user_profile = UserProfile.objects.get(license_plate=license_plate)
                user = user_profile.user
                if speed > SPEED_LIMIT:
                    # Create a violation record
                    violation = Violation.objects.create(
                        license_plate=user_profile,
                        vehicle_type="Car",
                        speed=speed
                    )
                    # Send a violation email to the user
                    send_violation_email(user, speed, violation.timestamp)
                    violations.append({
                        'license_plate': license_plate,
                        'speed': speed,
                        'timestamp': violation.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'email_sent': True
                    })
            except UserProfile.DoesNotExist:
                logger.warning(f"No user found for license plate: {license_plate}")
        
        # Return a success response with processed data
        return JsonResponse({
            'message': 'Video processed successfully',
            'file_url': fs.url(filename),
            'license_plates_and_speeds': license_plate_and_speeds,
            'violations': violations
        }, status=200)
    except ValidationError as e:
        logger.error(f"Video file validation failed: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Error handling video upload: {str(e)}")
        return JsonResponse({'error': 'Internal server error'}, status=500)

@csrf_exempt
def register_user(request) -> JsonResponse:
    """
    Handles user registration.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            license_plate = data.get('license_plate')

            # Validate input
            if not username or not email or not password or not license_plate:
                return JsonResponse({'error': 'All fields are required'}, status=400)

            # Check if the user already exists
            if User.objects.filter(username=username).exists():
                return JsonResponse({'error': 'Username already exists'}, status=400)

            # Create the user
            user = User.objects.create_user(username=username, email=email, password=password)
            UserProfile.objects.create(user=user, license_plate=license_plate)

            return JsonResponse({'message': 'User registered successfully'}, status=201)
        except Exception as e:
            logger.error(f"Error during user registration: {str(e)}")
            return JsonResponse({'error': 'Internal server error 1'}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)