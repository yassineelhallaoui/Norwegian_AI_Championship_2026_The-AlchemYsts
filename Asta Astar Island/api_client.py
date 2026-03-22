"""API client for Astar Island."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests


class AstarIslandClient:
    """Thin API wrapper with conservative retry handling."""

    def __init__(self, token: str, base_url: str = "https://api.ainm.no/astar-island") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Content-Type"] = "application/json"

    @classmethod
    def from_token_file(cls, token_file: str | Path) -> "AstarIslandClient":
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"No token found in {token_file}")
        return cls(token)

    def _get(self, path: str) -> Any:
        response = self.session.get(f"{self.base_url}/{path.lstrip('/')}", timeout=30)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any], max_retries: int = 5) -> Any:
        for attempt in range(max_retries):
            response = self.session.post(
                f"{self.base_url}/{path.lstrip('/')}",
                json=payload,
                timeout=60,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            wait_s = min(8.0, 0.5 * (2 ** attempt))
            time.sleep(wait_s)
        response.raise_for_status()
        return response.json()

    def get_rounds(self) -> list[dict[str, Any]]:
        return self._get("rounds")

    def get_round_details(self, round_id: str) -> dict[str, Any]:
        return self._get(f"rounds/{round_id}")

    def get_budget(self) -> dict[str, Any]:
        return self._get("budget")

    def get_my_rounds(self) -> list[dict[str, Any]]:
        return self._get("my-rounds")

    def get_my_predictions(self, round_id: str) -> list[dict[str, Any]]:
        return self._get(f"my-predictions/{round_id}")

    def get_leaderboard(self) -> list[dict[str, Any]]:
        return self._get("leaderboard")

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        return self._get(f"analysis/{round_id}/{seed_index}")

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> dict[str, Any]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        return self._post("simulate", payload)

    def submit_prediction(
        self,
        round_id: str,
        seed_index: int,
        prediction: list[list[list[float]]],
    ) -> dict[str, Any]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        return self._post("submit", payload)
