# patients/admin.py
from django.contrib import admin
from .models import Patient

@admin.action(description="Approve selected patients")
def approve_patients(modeladmin, request, queryset):
    updated = queryset.update(is_approved=True)
    modeladmin.message_user(request, f"{updated} patient(s) approved.")

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("user", "phone", "is_approved", "created_at")
    list_filter = ("is_approved", "created_at")
    search_fields = ("user__username", "user__email", "phone")
    actions = [approve_patients]
