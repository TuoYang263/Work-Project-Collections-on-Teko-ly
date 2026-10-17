"""Reliable local staging for monthly NYC TLC Yellow Taxi Parquet files.

Compatible with the existing Bronze call: download_to_local(year, months).
Optional expected_sha256 values must come from an independently trusted source.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts import config
from scripts.logger import get_logger

logger = get_logger()
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB
REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
}


def generate_url(year: int, month: int) -> str:
    if year < 2009 or not 1 <= month <= 12:
        raise ValueError(f"Invalid TLC data month: {year}-{month}")
    return (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"yellow_tripdata_{year}-{month:02d}.parquet"
    )


def _valid_parquet(path: Path) -> bool:
    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        return (
            metadata.num_rows > 0
            and metadata.num_row_groups > 0
            and REQUIRED_COLUMNS.issubset(parquet.schema_arrow.names)
        )
    except (OSError, ValueError, pa.ArrowInvalid):
        return False


def _sha256_hex(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_cached_file(path: Path, expected_digest: str | None) -> bool:
    if not path.is_file() or not _valid_parquet(path):
        return False
    return expected_digest is None or _sha256_hex(path) == expected_digest


@contextmanager
def _month_lock(directory: Path, filename: str) -> Iterator[None]:
    """Advisory lock for cooperating processes on the same Linux/WSL filesystem."""
    lock_dir = directory / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{filename}.lock"
    # Do not unlink the lock file: that could split simultaneous locks.
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _download_atomic(
    url: str,
    destination: Path,
    expected_digest: str | None,
) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            written = 0
            sha256 = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                length_header = response.headers.get("Content-Length")
                declared_length = (
                    int(length_header) if length_header is not None else None
                )
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                    if chunk:
                        temporary.write(chunk)
                        sha256.update(chunk)
                        written += len(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        if declared_length is not None and written != declared_length:
            raise ValueError(
                f"Incomplete HTTP download: expected={declared_length} actual={written}"
            )
        if expected_digest is not None and sha256.hexdigest() != expected_digest:
            raise ValueError("Trusted SHA-256 mismatch")
        if not _valid_parquet(temp_path):
            raise ValueError("HTTP download is not a valid NYC TLC Parquet")

        os.replace(temp_path, destination)
        logger.info("Validated raw Parquet staged: %s", destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def download_to_local(
    year: int,
    months: List[int],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> List[str]:
    """Download requested months and return only validated local file paths.

    expected_sha256 maps YYYY-MM to a trusted 64-character hexadecimal digest.
    No trusted digest available? Omit it; then only structural and size checks run.
    """
    directory = Path(config.TMP_PARQUET_PATH)
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    for month in months:
        url = generate_url(year, month)
        month_id = f"{year}-{month:02d}"
        filename = f"yellow_tripdata_{month_id}.parquet"
        destination = directory / filename

        digest: str | None = None
        if expected_sha256 is not None:
            if month_id not in expected_sha256:
                raise ValueError(f"Missing trusted SHA-256 for {month_id}")
            digest = expected_sha256[month_id].lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid trusted SHA-256 for {month_id}")

        # Recheck cache INSIDE lock to prevent duplicate concurrent downloads.
        with _month_lock(directory, filename):
            if _valid_cached_file(destination, digest):
                logger.info("Reusing validated local cache: %s", destination)
            else:
                logger.info("Downloading or replacing invalid cache: %s", url)
                _download_atomic(url, destination, digest)
            paths.append(str(destination))
    return paths


if __name__ == "__main__":
    logger.info("Files: %s", download_to_local(2023, [1, 2, 3]))
