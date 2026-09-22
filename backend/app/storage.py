from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import UploadFile

from .config import settings


class StorageError(ValueError):
    pass


def safe_filename(filename: str | None) -> str:
    candidate = Path(filename or "upload.bin").name
    candidate = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    return candidate or "upload.bin"


async def save_upload(upload: UploadFile, survey_id: int,
                      file_id: int) -> tuple[Path, int, str]:
    """Stream an upload to disk. Returns (path, bytes written, sha256).

    The digest is computed from the same chunks on the way past, so it costs
    nothing extra and the file never has to be read back to identify it.
    """
    filename = safe_filename(upload.filename)
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        await upload.close()
        raise StorageError(f"Unsupported file type. Allowed extensions: {allowed}")

    destination_dir = settings.uploads_dir / str(survey_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{file_id}_{filename}"
    total = 0
    digest = hashlib.sha256()

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_size_bytes:
                    raise StorageError(
                        f"File is too large. Maximum size is {settings.max_upload_size_bytes} bytes"
                    )
                digest.update(chunk)
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise StorageError("Uploaded file is empty")

    return destination, total, digest.hexdigest()
