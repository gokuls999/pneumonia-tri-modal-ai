from django.urls import path
from .views import PredictView, PredictPageView

urlpatterns = [
    path("", PredictPageView.as_view(), name="predict-ui"),
    path("predict/", PredictView.as_view(), name="predict"),
]
