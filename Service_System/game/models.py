import uuid

from django.conf import settings
from django.db import models


class GameSession(models.Model):
    class Status(models.TextChoices):
        READY = "READY", "Ready"
        PROCESSING = "PROCESSING", "Processing"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="game_sessions")
    before_image = models.ImageField(upload_to="game/before/%Y/%m/%d/")
    after_image = models.ImageField(upload_to="game/after/%Y/%m/%d/")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.READY, db_index=True)
    assigned_worker = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="claimed_game_sessions"
    )
    before_counts = models.JSONField(default=dict, blank=True)
    after_counts = models.JSONField(default=dict, blank=True)
    difference = models.JSONField(default=dict, blank=True)
    error_message = models.CharField(max_length=500, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    claimed_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.pk} ({self.status})"
