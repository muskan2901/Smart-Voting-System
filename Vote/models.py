from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    voted = models.BooleanField(default=False)
    id = models.CharField(max_length=100, primary_key=True)

    def __str__(self):
        return self.user.username


class Position(models.Model):
    position = models.CharField(max_length=50)
    no_of_candidates = models.IntegerField(default=0)
    about = models.TextField(default='')

    def __str__(self):
        return self.position


class Candidate(models.Model):
    candidate = models.ForeignKey(Position, on_delete=models.CASCADE)

    name = models.CharField(max_length=50)
    Description = models.TextField()
    image = models.ImageField(upload_to='Vote/static/Vote', blank=True)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.name
