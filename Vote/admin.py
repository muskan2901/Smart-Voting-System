from django.contrib import admin
from .models import Position, Candidate, UserProfile

admin.site.register(UserProfile)
admin.site.register(Position)
admin.site.register(Candidate)

