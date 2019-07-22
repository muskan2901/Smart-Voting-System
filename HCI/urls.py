from django.contrib import admin
from django.conf.urls import include, url
from django.urls import path

urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^Vote/', include('Vote.urls')),
    path('', include('Vote.urls')),
    url(r'^records/', include('records.urls')),
]
