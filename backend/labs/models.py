from django.db import models

# Create your models here.
class ExampleModel(models.Model):
    name = models.CharField(max_length=100)

class Lab(models.Model):  # Add this class
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name
