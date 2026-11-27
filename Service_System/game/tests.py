from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from .models import GameSession


def picture(name):
    stream = BytesIO()
    Image.new("RGB", (32, 32), "red").save(stream, format="PNG")
    return SimpleUploadedFile(name, stream.getvalue(), content_type="image/png")


class GameAPITests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="player", password="not-used")
        self.other = User.objects.create_user(username="other", password="not-used")
        self.worker = User.objects.create_user(username="edge_worker", password="not-used", is_staff=True)
        self.player_api = APIClient()
        self.player_api.credentials(HTTP_AUTHORIZATION=f"Token {Token.objects.create(user=self.user).key}")
        self.other_api = APIClient()
        self.other_api.credentials(HTTP_AUTHORIZATION=f"Token {Token.objects.create(user=self.other).key}")
        self.worker_api = APIClient()
        self.worker_api.credentials(HTTP_AUTHORIZATION=f"Token {Token.objects.create(user=self.worker).key}")

    def create_game(self):
        result = self.player_api.post(
            "/api/game/",
            {"before_image": picture("before.png"), "after_image": picture("after.png")},
            format="multipart",
        )
        self.assertEqual(result.status_code, 201, result.data)
        return result.data["id"]

    def test_game_create_claim_complete_and_read(self):
        game_id = self.create_game()
        self.assertEqual(GameSession.objects.get(pk=game_id).status, "READY")
        self.assertEqual(self.other_api.get(f"/api/game/{game_id}/").status_code, 404)
        self.assertEqual(self.player_api.post("/api/game/worker/claim/").status_code, 403)
        claim = self.worker_api.post("/api/game/worker/claim/")
        self.assertEqual(claim.status_code, 200)
        self.assertEqual(str(claim.data["id"]), str(game_id))
        self.assertEqual(self.worker_api.post("/api/game/worker/claim/").status_code, 204)
        self.assertEqual(self.worker_api.get(claim.data["before_download"]).status_code, 200)
        self.assertEqual(self.other_api.get(claim.data["before_download"]).status_code, 403)
        result = self.worker_api.post(
            f"/api/game/worker/{game_id}/complete/",
            {"before_counts": {"bottle": 1}, "after_counts": {"book": 1},
             "difference": {"added": {"book": 1}, "removed": {"bottle": 1}}},
            format="json",
        )
        self.assertEqual(result.status_code, 200, result.data)
        detail = self.player_api.get(f"/api/game/{game_id}/")
        self.assertEqual(detail.data["status"], "COMPLETED")
        self.assertEqual(detail.data["difference"]["added"], {"book": 1})
        self.assertEqual(self.worker_api.post(f"/api/game/worker/{game_id}/complete/", {}, format="json").status_code, 400)

    def test_reject_invalid_difference(self):
        game_id = self.create_game()
        self.worker_api.post("/api/game/worker/claim/")
        result = self.worker_api.post(
            f"/api/game/worker/{game_id}/complete/",
            {"before_counts": {"bottle": 1}, "after_counts": {"bottle": 0}, "difference": {"added": {}, "removed": {}}},
            format="json",
        )
        self.assertEqual(result.status_code, 400)
        self.assertEqual(GameSession.objects.get(pk=game_id).status, "PROCESSING")

    def test_anonymous_cannot_create_game(self):
        self.assertEqual(APIClient().get("/api/game/").status_code, 401)
