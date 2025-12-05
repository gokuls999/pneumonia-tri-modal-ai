from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("created_at", "predicted_class", "confidence")
    list_filter = ("predicted_class", "created_at")
    search_fields = ("clinical_text",)
