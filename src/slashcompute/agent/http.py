"""HTTP client for dataset / checkpoint / verification blobs on the coordinator."""

from __future__ import annotations

from pathlib import Path

import httpx


class CoordHTTP:
    def __init__(self, base: str, timeout: float = 120.0, session_token: str | None = None) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session_token = session_token

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return self.base + (path if path.startswith("/") else "/" + path)

    def _headers(self, url: str) -> dict[str, str]:
        base, target = httpx.URL(self.base), httpx.URL(url)
        # Assignments may refer to external datasets; never send our session there.
        if self.session_token and (base.scheme, base.host, base.port) == (
            target.scheme, target.host, target.port,
        ):
            return {"Authorization": f"Bearer {self.session_token}"}
        return {}

    def get_file(self, path: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self.url(path)
        with httpx.stream("GET", url, headers=self._headers(url), timeout=self.timeout) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest

    def put_bytes(self, path: str, data: bytes, params: dict | None = None) -> None:
        url = self.url(path)
        r = httpx.post(url, content=data, params=params, headers=self._headers(url), timeout=self.timeout)
        r.raise_for_status()
