from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    full_name = models.CharField(max_length=150)
    age = models.IntegerField(null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)

    is_approved = models.BooleanField(default=False)   # Admin will approve

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.full_name
