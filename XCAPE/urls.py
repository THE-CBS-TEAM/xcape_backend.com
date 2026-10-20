"""
URL configuration for XCAPE project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Django Admin Panel Custom Branding
admin.site.site_header = "XCAPE Reservoir Simulation & History Matching Admin"
admin.site.site_title = "XCAPE Admin Portal"
admin.site.index_title = "Platform Operations, User Management & Simulation Data"
