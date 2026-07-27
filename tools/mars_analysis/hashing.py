from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    exists: bool
    size: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None

    def stable_value(self) -> str:
        return "|".join(
            [
                self.path,
                str(self.exists),
                str(self.size),
                str(self.mtime_ns),
                str(self.sha256),
            ]
        )


def file_fingerprint(path: str | Path, *, content_hash: bool = False) -> FileFingerprint:
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return FileFingerprint(path=str(path), exists=False)
    sha = sha256_file(path) if content_hash else None
    return FileFingerprint(
        path=str(path),
        exists=True,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha,
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]
