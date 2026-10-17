"""Local-only raw staging tests: no real HTTP calls or AWS charges."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from threading import Event, Lock

import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from scripts import load_nyc_yellow_taxi_data as downloader


def _parquet_bytes(row_count=1) -> bytes:
    table = pa.table(
        {
            "tpep_pickup_datetime": [datetime(2024, 1, 1, 10)] * row_count,
            "tpep_dropoff_datetime": [datetime(2024, 1, 1, 10, 20)] * row_count,
            "PULocationID": [100] * row_count,
            "DOLocationID": [200] * row_count,
        }
    )
    output = BytesIO()
    pq.write_table(table, output)
    return output.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, *, content_length: int | None = None):
        self.content = content
        self.headers = {
            "Content-Length": str(
                len(content) if content_length is None else content_length
            )
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass  # HTTP 200 OK

    def iter_content(self, chunk_size):
        # Yield two chunks to exercise the streaming code.
        midpoint = max(1, len(self.content) // 2)
        yield self.content[:midpoint]
        yield self.content[midpoint:]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader.config, "TMP_PARQUET_PATH", str(tmp_path))
    return tmp_path


def _target(directory):
    return directory / "yellow_tripdata_2024-01.parquet"


def _no_temp_parts(directory):
    assert list(directory.glob("*.part")) == []
    assert list(directory.glob(".*.part")) == []


def test_corrupted_cached_file_is_redownloaded(cache_dir, monkeypatch):
    destination = _target(cache_dir)
    destination.write_bytes(b"corrupted parquet")
    valid = _parquet_bytes()
    requests_made = []

    def fake_get(url, **kwargs):
        requests_made.append(url)
        return FakeResponse(valid)

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    paths = downloader.download_to_local(2024, [1])

    assert paths == [str(destination)]
    assert len(requests_made) == 1
    assert destination.read_bytes() == valid
    assert pq.ParquetFile(destination).metadata.num_rows == 1
    _no_temp_parts(cache_dir)


def test_interrupted_download_cleans_temporary_file(cache_dir, monkeypatch):
    class BrokenResponse(FakeResponse):
        def iter_content(self, chunk_size):
            yield b"incomplete parquet bytes"
            raise requests.exceptions.ChunkedEncodingError("Connection interrupted")

    monkeypatch.setattr(
        downloader.requests, "get", lambda *a, **kw: BrokenResponse(b"unused")
    )
    with pytest.raises(
        requests.exceptions.ChunkedEncodingError, match="Connection interrupted"
    ):
        downloader.download_to_local(2024, [1])

    assert not _target(cache_dir).exists()
    _no_temp_parts(cache_dir)
    # A persistent .locks directory is intentional and safe.


def test_valid_cached_file_is_reused(cache_dir, monkeypatch):
    destination = _target(cache_dir)
    valid = _parquet_bytes()
    destination.write_bytes(valid)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Valid cache must not trigger HTTP")

    monkeypatch.setattr(downloader.requests, "get", unexpected_request)
    assert downloader.download_to_local(2024, [1]) == [str(destination)]
    assert destination.read_bytes() == valid


def test_http_200_invalid_parquet_is_rejected(cache_dir, monkeypatch):
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda *a, **kw: FakeResponse(b"HTTP 200 but not parquet"),
    )
    with pytest.raises(ValueError, match="not a valid NYC TLC Parquet"):
        downloader.download_to_local(2024, [1])

    assert not _target(cache_dir).exists()
    _no_temp_parts(cache_dir)


def test_trusted_sha256_mismatch_rejects_valid_parquet(cache_dir, monkeypatch):
    valid = _parquet_bytes()
    trusted_digest = hashlib.sha256(b"different trusted bytes").hexdigest()
    monkeypatch.setattr(
        downloader.requests, "get", lambda *a, **kw: FakeResponse(valid)
    )
    with pytest.raises(ValueError, match="Trusted SHA-256 mismatch"):
        downloader.download_to_local(
            2024, [1], expected_sha256={"2024-01": trusted_digest}
        )

    assert not _target(cache_dir).exists()
    _no_temp_parts(cache_dir)


def test_valid_parquet_with_wrong_cached_digest_is_replaced(cache_dir, monkeypatch):
    destination = _target(cache_dir)
    destination.write_bytes(_parquet_bytes(row_count=2))
    expected = _parquet_bytes(row_count=1)
    digest = hashlib.sha256(expected).hexdigest()
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(expected)

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    assert downloader.download_to_local(
        2024, [1], expected_sha256={"2024-01": digest}
    ) == [str(destination)]
    assert len(calls) == 1
    assert destination.read_bytes() == expected
    _no_temp_parts(cache_dir)


def test_http_content_length_mismatch_rejected(cache_dir, monkeypatch):
    valid = _parquet_bytes()
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda *a, **kw: FakeResponse(valid, content_length=len(valid) + 10),
    )
    with pytest.raises(ValueError, match="Incomplete HTTP download"):
        downloader.download_to_local(2024, [1])

    assert not _target(cache_dir).exists()
    _no_temp_parts(cache_dir)


def test_concurrent_downloads_publish_once(cache_dir, monkeypatch):
    content = _parquet_bytes()
    first_request_started = Event()
    release_first_download = Event()
    counter_lock = Lock()
    calls = []

    class SlowResponse(FakeResponse):
        def iter_content(self, chunk_size):
            first_request_started.set()
            if not release_first_download.wait(timeout=10):
                raise TimeoutError("Test download gate timed out")
            yield from super().iter_content(chunk_size)

    def fake_get(url, **kwargs):
        with counter_lock:
            calls.append(url)
        return SlowResponse(content)

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_one = pool.submit(downloader.download_to_local, 2024, [1])
        assert first_request_started.wait(timeout=10)
        future_two = pool.submit(downloader.download_to_local, 2024, [1])
        release_first_download.set()
        result_one = future_one.result(timeout=15)
        result_two = future_two.result(timeout=15)

    assert result_one == result_two == [str(_target(cache_dir))]
    assert len(calls) == 1
    assert _target(cache_dir).read_bytes() == content
    _no_temp_parts(cache_dir)
