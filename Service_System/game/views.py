from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import generics, serializers, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import GameSession
from .serializers import GameSessionCreateSerializer, GameSessionReadSerializer


class GameListCreateView(generics.ListCreateAPIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return GameSession.objects.filter(owner=self.request.user)

    def get_serializer_class(self):
        if self.request.method == "POST":
            return GameSessionCreateSerializer
        return GameSessionReadSerializer

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class GameDetailView(generics.RetrieveAPIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    serializer_class = GameSessionReadSerializer

    def get_queryset(self):
        return GameSession.objects.filter(owner=self.request.user)


def _require_worker(request):
    # Separate worker identity: issue a token for a dedicated staff account.
    if not request.user.is_staff:
        return Response({"detail": "Worker permission required."}, status=status.HTTP_403_FORBIDDEN)
    return None


def _valid_counts(value):
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(name, str) and bool(name) and type(count) is int and count >= 0
        for name, count in value.items()
    )


def _expected_difference(before, after):
    names = set(before) | set(after)
    return {
        "added": {name: after.get(name, 0) - before.get(name, 0) for name in sorted(names) if after.get(name, 0) > before.get(name, 0)},
        "removed": {name: before.get(name, 0) - after.get(name, 0) for name in sorted(names) if before.get(name, 0) > after.get(name, 0)},
    }


@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def worker_claim(request):
    denied = _require_worker(request)
    if denied is not None:
        return denied

    # Atomic conditional update works with SQLite as well as PostgreSQL.
    # Multiple workers racing to claim the same item cannot both succeed.
    while True:
        candidate = GameSession.objects.filter(status=GameSession.Status.READY).order_by("created_at", "pk").values_list("pk", flat=True).first()
        if candidate is None:
            return Response(status=status.HTTP_204_NO_CONTENT)
        updated = GameSession.objects.filter(pk=candidate, status=GameSession.Status.READY).update(
            status=GameSession.Status.PROCESSING,
            assigned_worker=request.user,
            claimed_at=timezone.now(),
        )
        if updated:
            return Response({
                "id": str(candidate),
                "before_download": f"/api/game/worker/{candidate}/image/before/",
                "after_download": f"/api/game/worker/{candidate}/image/after/",
            }, status=status.HTTP_200_OK)


@api_view(["GET"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def worker_image(request, pk, kind):
    denied = _require_worker(request)
    if denied is not None:
        return denied
    if kind not in ("before", "after"):
        raise Http404("Unknown image type")
    game = get_object_or_404(GameSession, pk=pk, status=GameSession.Status.PROCESSING, assigned_worker=request.user)
    image = game.before_image if kind == "before" else game.after_image
    if not image:
        raise Http404("Image missing")
    try:
        image.open("rb")
    except (OSError, ValueError):
        raise Http404("Image not available")
    return FileResponse(image, content_type="application/octet-stream", as_attachment=True, filename=f"{kind}.jpg")


@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def worker_complete(request, pk):
    denied = _require_worker(request)
    if denied is not None:
        return denied
    before = request.data.get("before_counts")
    after = request.data.get("after_counts")
    difference = request.data.get("difference")
    if not _valid_counts(before) or not _valid_counts(after):
        return Response({"detail": "Invalid object counts."}, status=status.HTTP_400_BAD_REQUEST)
    expected = _expected_difference(before, after)
    if difference != expected:
        return Response({"detail": "Difference does not match supplied counts."}, status=status.HTTP_400_BAD_REQUEST)

    updated = GameSession.objects.filter(
        pk=pk, status=GameSession.Status.PROCESSING, assigned_worker=request.user
    ).update(
        status=GameSession.Status.COMPLETED,
        before_counts=before,
        after_counts=after,
        difference=expected,
        error_message="",
        completed_at=timezone.now(),
    )
    if not updated:
        return Response({"detail": "Game not claimed by this worker or already processed."}, status=status.HTTP_409_CONFLICT)
    return Response({"id": str(pk), "status": GameSession.Status.COMPLETED})


@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def worker_fail(request, pk):
    denied = _require_worker(request)
    if denied is not None:
        return denied
    reason = request.data.get("error_message", "Analysis failed")
    if not isinstance(reason, str):
        raise serializers.ValidationError({"error_message": "Expected string."})
    # Don't leak machine paths or stack traces to the Android client.
    updated = GameSession.objects.filter(
        pk=pk, status=GameSession.Status.PROCESSING, assigned_worker=request.user
    ).update(
        status=GameSession.Status.FAILED,
        error_message=reason[:500],
        completed_at=timezone.now(),
    )
    if not updated:
        return Response({"detail": "Game not claimed by this worker or already processed."}, status=status.HTTP_409_CONFLICT)
    return Response({"id": str(pk), "status": GameSession.Status.FAILED})
