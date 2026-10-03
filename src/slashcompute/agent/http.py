"""HTTP client for dataset / checkpoint / verification blobs on the coordinator."""

from __future__ import annotations

from pathlib import Path

import httpx


class CoordHTTP:
    def __init__(self, base: str, timeout: float = 120.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return self.base + (path if path.startswith("/") else "/" + path)

    def get_file(self, path: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", self.url(path), timeout=self.timeout) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest

    def put_bytes(self, path: str, data: bytes, params: dict | None = None) -> None:
        r = httpx.post(self.url(path), content=data, params=params, timeout=self.timeout)
        r.raise_for_status()
