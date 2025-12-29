# pneumonia_api/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# import the three endpoints from api.views_tri
from api.views_tri import (
    tri_infer_view,
    tri_infer_upload_view,
    generate_report_view,
)

urlpatterns = [
    path("admin/", admin.site.urls),

    # API endpoints
    path("api/tri-infer/", tri_infer_view, name="tri-infer"),                  # existing JSON POST
    path("api/tri-infer-upload/", tri_infer_upload_view, name="tri-infer-upload"),  # multipart upload
    path("api/generate-report/", generate_report_view, name="generate-report"),
    path("patients/", include("patients.urls", namespace="patients")),     # PDF generator

    # Root URL -> UI (your simple index / predictor app)
    path("", include("predictor.urls")),
]

# Serve media files in development (only)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
