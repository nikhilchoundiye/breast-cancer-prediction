from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('labs/', include('labs.urls')),  # This exists in your project
    path('api/', include('labs.urls')),  # Add this line (Replace `your_app` with your actual Django app name)
]



