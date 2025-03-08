
# from django.contrib import admin
# from django.urls import path, include

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('vehicle/', include('vehicle_processing.urls')),
# ]
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin URL
    path('api/', include('vehicle_processing.urls')),  # Include app URLs under /api/
]