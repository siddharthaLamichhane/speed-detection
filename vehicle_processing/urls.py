# vehicle_processing/urls.py
# from django.urls import path
# from . import views

# urlpatterns = [
#     path('process/', views.process_video, name='process_video'),
# ]

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('register/', views.register_user, name='register_user'),
#     path('upload/', views.upload_video, name='upload_video'),
# ]
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_user, name='register_user'),
    path('upload/', views.upload_video, name='upload_video'),
]
