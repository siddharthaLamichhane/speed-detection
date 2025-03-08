# from django.db import models
# from django.core.validators import RegexValidator

# # License plate validation (optional)
# license_plate_validator = RegexValidator(
#     regex=r'^[A-Za-z0-9]{1,20}$',  # Adjust this pattern to match your license plate format
#     message="Enter a valid license plate."
# )

# class User(models.Model):
#     username = models.CharField(max_length=100)
#     email = models.EmailField(unique=True)
#     license_plate = models.CharField(
#         max_length=20,
#         unique=True,
#         validators=[license_plate_validator]
#     )

#     def __str__(self):
#         return self.username

# class Violation(models.Model):
#     license_plate = models.ForeignKey(User, on_delete=models.CASCADE)
#     vehicle_type = models.CharField(max_length=50)
#     speed = models.FloatField()
#     timestamp = models.DateTimeField(auto_now_add=True)

#     class Meta:
#         indexes = [
#             models.Index(fields=['license_plate']),
#             models.Index(fields=['timestamp']),
#         ]

#     def __str__(self):
#         return f"{self.vehicle_type} - {self.license_plate} - {self.speed} km/h"


from django.db import models
from django.contrib.auth.models import User
from django.core.validators import RegexValidator, MinValueValidator
from django.utils.translation import gettext_lazy as _

# License plate validation (optional)
license_plate_validator = RegexValidator(
    regex=r'^[A-Za-z0-9]{1,20}$',  # Adjust this pattern to match your license plate format
    message=_("Enter a valid license plate (alphanumeric, 1-20 characters).")
)

class UserProfile(models.Model):
    """
    Extends the User model to store additional information like license plate.
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
        verbose_name=_("User"),
        help_text=_("The user associated with this profile.")
    )
    license_plate = models.CharField(
        max_length=20,
        unique=True,
        validators=[license_plate_validator],
        verbose_name=_("License Plate"),
        help_text=_("The license plate of the user's vehicle.")
    )

    def __str__(self):
        return f"{self.user.username} - {self.license_plate}"

    class Meta:
        verbose_name = _("User Profile")
        verbose_name_plural = _("User Profiles")
        ordering = ["user__username"]  # Order by username


class Violation(models.Model):
    """
    Represents a speed violation recorded for a user's vehicle.
    """
    license_plate = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="violations",
        verbose_name=_("License Plate"),
        help_text=_("The user associated with this violation.")
    )
    vehicle_type = models.CharField(
        max_length=50,
        verbose_name=_("Vehicle Type"),
        help_text=_("The type of vehicle involved in the violation.")
    )
    speed = models.FloatField(
        verbose_name=_("Speed"),
        help_text=_("The recorded speed of the vehicle in km/h."),
        validators=[MinValueValidator(0)]  # Ensure speed is not negative
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_("Timestamp"),
        help_text=_("The date and time when the violation was recorded.")
    )

    def __str__(self):
        return f"{self.vehicle_type} - {self.license_plate} - {self.speed} km/h"

    class Meta:
        verbose_name = _("Violation")
        verbose_name_plural = _("Violations")
        indexes = [
            models.Index(fields=['license_plate']),
            models.Index(fields=['timestamp']),
        ]
        ordering = ["-timestamp"]  # Order violations by most recent first