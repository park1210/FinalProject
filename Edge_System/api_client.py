import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


class GameAPIError(RuntimeError):
    pass


class GameAPIClient:
    def __init__(self, base_url=None, token=None, timeout=30):
        self.base_url = (base_url or os.getenv("GAME_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/") + "/"
        self.token = token or os.getenv("GAME_WORKER_TOKEN")
        if not self.token:
            raise ValueError("GAME_WORKER_TOKEN is required")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {self.token}"})

    def _url(self, path):
        # Only server-returned relative paths are accepted.
        parsed = urlparse(path)
        if parsed.scheme or parsed.netloc or not path.startswith("/api/game/worker/"):
            raise GameAPIError("Invalid worker API path")
        return urljoin(self.base_url, path.lstrip("/"))

    def _request(self, method, path, **kwargs):
        try:
            response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise GameAPIError(f"API request failed: {exc}") from exc

    def claim(self):
        response = self._request("POST", "/api/game/worker/claim/")
        if response.status_code == 204:
            return None
        return response.json()

    def download(self, relative_url, target_path):
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._request("GET", relative_url, stream=True) as response:
                with path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            output.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def complete(self, game_id, before_counts, after_counts, difference):
        response = self._request(
            "POST", f"/api/game/worker/{game_id}/complete/",
            json={"before_counts": before_counts, "after_counts": after_counts, "difference": difference}
        )
        return response.json()

    def fail(self, game_id, message="Analysis failed"):
        response = self._request(
            "POST", f"/api/game/worker/{game_id}/fail/",
            json={"error_message": message}
        )
        return response.json()
