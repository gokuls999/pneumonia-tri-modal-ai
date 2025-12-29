from django import forms
from django.contrib.auth.models import User
from .models import Patient


class PatientRegistrationForm(forms.ModelForm):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
    email = forms.EmailField()

    class Meta:
        model = Patient
        fields = ["full_name", "age", "phone", "address"]

    def save(self, commit=True):
        # Create Django user first
        user = User.objects.create_user(
            username=self.cleaned_data["username"],
            password=self.cleaned_data["password"],
            email=self.cleaned_data["email"],
        )

        patient = super().save(commit=False)
        patient.user = user
        if commit:
            patient.save()
        return patient
