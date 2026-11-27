from django.contrib import admin

from .models import GameSession


@admin.register(GameSession)
class GameSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "status", "assigned_worker", "created_at", "completed_at")
    list_filter = ("status", "created_at")
    search_fields = ("owner__username", "id")
    readonly_fields = ("created_at", "claimed_at", "completed_at")
