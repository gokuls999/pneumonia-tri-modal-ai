# patients/urls.py
from django.urls import path
from . import views

app_name = "patients"

urlpatterns = [
    path("register/", views.register_patient, name="register"),
    path("login/", views.login_patient, name="login"),
    path("logout/", views.logout_patient, name="logout"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
