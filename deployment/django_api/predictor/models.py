from django.db import models


class Prediction(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    xray_image = models.ImageField(upload_to="xray_uploads/")
    ct_image = models.ImageField(upload_to="ct_uploads/", blank=True, null=True)
    clinical_text = models.TextField()

    predicted_class = models.CharField(max_length=20)
    predicted_class = models.CharField(max_length=20)
    confidence = models.FloatField()

    gradcam_image = models.ImageField(upload_to="gradcam/", blank=True, null=True)

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2f}) at {self.created_at}"
