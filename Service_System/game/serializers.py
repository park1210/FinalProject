from rest_framework import serializers

from .models import GameSession


class GameSessionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameSession
        fields = ("id", "before_image", "after_image", "status", "created_at")
        read_only_fields = ("id", "status", "created_at")

    def validate(self, attrs):
        for key in ("before_image", "after_image"):
            uploaded = attrs.get(key)
            if uploaded is None:
                raise serializers.ValidationError({key: "Image is required."})
            if uploaded.size > 10 * 1024 * 1024:
                raise serializers.ValidationError({key: "Image must be 10 MB or smaller."})
        return attrs


class GameSessionReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameSession
        fields = (
            "id", "status", "created_at", "claimed_at", "completed_at",
            "before_counts", "after_counts", "difference", "error_message",
        )
        read_only_fields = fields
