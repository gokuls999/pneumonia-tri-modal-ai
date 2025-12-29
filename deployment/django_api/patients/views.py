# patients/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.urls import reverse


def register_patient(request):
    """
    Show registration form and create a new Django user for the patient.
    Template expected: patients/register.html
    """
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Optionally auto-login after register
            login(request, user)
            messages.success(request, "Registration successful. You are now logged in.")
            return redirect(reverse("patients:dashboard"))
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserCreationForm()

    return render(request, "patients/register.html", {"form": form})


def login_patient(request):
    """
    Patient login view.
    Template expected: patients/login.html
    """
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "Logged in successfully.")
            # Redirect to 'next' if present
            next_url = request.GET.get("next") or request.POST.get("next")
            if next_url:
                return redirect(next_url)
            return redirect(reverse("patients:dashboard"))
        else:
            messages.error(request, "Invalid credentials. Please try again.")
    else:
        form = AuthenticationForm(request)

    return render(request, "patients/login.html", {"form": form})


@login_required
def logout_patient(request):
    """
    Log the current patient out.
    """
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect(reverse("patients:login"))


@login_required
def dashboard(request):
    """
    Simple patient dashboard placeholder.
    Template expected: patients/dashboard.html
    """
    # You can expand 'context' later with patient-specific data
    context = {
        "user": request.user,
    }
    return render(request, "patients/dashboard.html", context)
