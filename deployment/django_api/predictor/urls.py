from django.urls import path
from .simple_views import SimpleTriPageView

urlpatterns = [
    path("", SimpleTriPageView.as_view(), name="tri-page"),
]
