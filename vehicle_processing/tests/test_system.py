import os
import json
from django.test import TestCase, Client
from django.core import mail
from django.conf import settings
from ..models import User, Violation

class SystemTest(TestCase):
    def setUp(self):
        self.client = Client()
        # Create test user
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'license_plate': 'ABC123'
        }

    def test_user_registration(self):
        """Test user registration endpoint"""
        response = self.client.post('/api/register/',
                                  data=json.dumps(self.user_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 201)
        self.assertTrue(User.objects.filter(email=self.user_data['email']).exists())

    def test_video_upload_and_violation(self):
        """Test video upload and violation detection"""
        # First register a user
        user = User.objects.create(**self.user_data)
        
        # Create a test video file path
        video_path = os.path.join(settings.MEDIA_ROOT, 'test_video.mp4')
        
        # Test video upload
        with open(video_path, 'rb') if os.path.exists(video_path) else None as video:
            if video:
                response = self.client.post('/api/upload/',
                                          {'video': video},
                                          format='multipart')
                self.assertEqual(response.status_code, 200)
                
                # Check if violation was created
                self.assertTrue(Violation.objects.filter(license_plate=user).exists())
                
                # Check if email was sent
                self.assertEqual(len(mail.outbox), 1)
                self.assertIn(user.email, mail.outbox[0].to)
            else:
                print("Note: Skipping video upload test as test video file not found") 