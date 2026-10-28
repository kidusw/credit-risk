"""Small wrapper around the crisk HTTP API, used by the dashboard (httpx only)."""

import os

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8001"


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0, http=None):
        """`http` lets tests pass a FastAPI TestClient (itself an httpx.Client)."""
        base_url = base_url or os.environ.get("CRISK_API_URL", DEFAULT_API_URL)
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url} ({exc})") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{method} {path} failed ({response.status_code}): {detail}")
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def models(self) -> dict:
        return self._request("GET", "/model")

    def examples(self, track: str) -> list[dict]:
        return self._request("GET", f"/examples/{track}")

    def score(self, track: str, applicant: dict, ref: str | None = None) -> dict:
        body = {"track": track, "applicant": applicant, "applicant_ref": ref}
        return self._request("POST", "/score", json=body)

    def decisions(self, limit: int = 20) -> list[dict]:
        return self._request("GET", "/decisions", params={"limit": limit})
