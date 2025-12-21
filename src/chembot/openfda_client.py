from __future__ import annotations
import requests
from typing import Any, Dict, List, Optional

class OpenFDAClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.api_key:
            params = dict(params)
            params["api_key"] = self.api_key
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def search_labels(self, search: str, limit: int = 5, skip: int = 0) -> List[Dict[str, Any]]:
        # https://api.fda.gov/drug/label.json?search=...&limit=...&skip=...
        url = "https://api.fda.gov/drug/label.json"
        data = self._get(url, {"search": search, "limit": limit, "skip": skip})
        return data.get("results", [])

    def search_events(self, search: str, limit: int = 5, skip: int = 0) -> List[Dict[str, Any]]:
        # https://api.fda.gov/drug/event.json?search=...&limit=...&skip=...
        url = "https://api.fda.gov/drug/event.json"
        data = self._get(url, {"search": search, "limit": limit, "skip": skip})
        return data.get("results", [])