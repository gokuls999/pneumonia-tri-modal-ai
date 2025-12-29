# deployment/django_api/predictor/simple_views.py

from django.views.generic import TemplateView

class SimpleTriPageView(TemplateView):
    """
    Very simple page that just renders templates/predictor/index.html
    and lets JavaScript call /api/tri-infer/ directly.
    """
    template_name = "predictor/index.html"
