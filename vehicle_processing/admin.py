from django.contrib import admin
from .models import UserProfile, Violation

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'license_plate')
    search_fields = ('user__username', 'license_plate')
    list_filter = ('user__username',)

@admin.register(Violation)
class ViolationAdmin(admin.ModelAdmin):
    list_display = ('license_plate', 'vehicle_type', 'speed', 'timestamp')
    search_fields = ('license_plate__license_plate', 'vehicle_type')
    list_filter = ('vehicle_type', 'timestamp')
    date_hierarchy = 'timestamp'