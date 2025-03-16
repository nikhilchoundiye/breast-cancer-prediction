from django.urls import path
from .views import validate_license  # Ensure this is correctly imported

urlpatterns = [
    path("validate_license/", validate_license, name="validate_license"),  # ✅ Use dash (-) if that's your convention
]
