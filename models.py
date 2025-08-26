from django.db import models

class Lab(models.Model):
    license_number = models.CharField(max_length=50, unique=True)
    expiration_date = models.DateField()
    status = models.CharField(max_length=20, default="Pending")  # Auto-updated to "Approved" or "Rejected"
