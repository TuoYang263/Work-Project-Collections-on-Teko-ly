import json
import base64
import hashlib
from io import BytesIO
from pathlib import Path
from datetime import datetime

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import s3_raw_reader
from scripts.s3_raw_ingestion import build_raw_manifest
from scripts.s3_raw_reader import (
    read_raw_manifest,
    download_versioned_raw,
)


def sample_reference():
    return {
        "data_month": "2024-01",
        "bucket": "test-bucket",
        "raw_key": "raw/nyc_tlc/yellow/2024/01/yellow_tripdata_2024-01.parquet",
        "raw_version_id": "raw-v1",
        "manifest_key": "raw/nyc_tlc/yellow/2024/01/yellow_tripdata_2024-01.parquet.manifest.json",
    }


def test_read_raw_manifest_matches_version():
    reference = sample_reference()

    manifest = build_raw_manifest(
        year=2024,
        month=1,
        bucket="test-bucket",
        ingestion_result={
            "status": "REUSED",
            "metadata": {
                "size": 100,
                "checksum_sha256": "example-checksum",
                "version_id": "raw-v1",
                "encryption": "AES256",
            },
        },
    )

    class FakeS3:
        def get_object(self, **kwargs):
            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == reference["manifest_key"]

            return {"Body": BytesIO(json.dumps(manifest).encode())}

    result = read_raw_manifest(FakeS3(), reference)

    assert result["s3"]["version_id"] == "raw-v1"
    assert result["data_month"] == "2024-01"


def test_read_raw_manifest_rejects_wrong_version():
    reference = sample_reference()

    manifest = build_raw_manifest(
        year=2024,
        month=1,
        bucket="test-bucket",
        ingestion_result={
            "status": "REUSED",
            "metadata": {
                "size": 100,
                "checksum_sha256": "example-checksum",
                "version_id": "raw-v2",
                "encryption": "AES256",
            },
        },
    )

    class FakeS3:
        def get_object(self, **kwargs):
            return {"Body": BytesIO(json.dumps(manifest).encode())}

    with pytest.raises(ValueError, match="requested raw version"):
        read_raw_manifest(FakeS3(), reference)


@pytest.mark.parametrize("bad_checksum", [False, True])
def test_download_versioned_raw(
    tmp_path,
    bad_checksum,
):
    reference = sample_reference()

    table = pa.table(
        {
            "tpep_pickup_datetime": [datetime(2024, 1, 1, 10)],
            "tpep_dropoff_datetime": [datetime(2024, 1, 1, 10, 20)],
            "PULocationID": [100],
            "DOLocationID": [200],
        }
    )

    output = BytesIO()
    pq.write_table(table, output)
    raw_bytes = output.getvalue()

    checksum = base64.b64encode(hashlib.sha256(raw_bytes).digest()).decode("ascii")

    manifest = build_raw_manifest(
        year=2024,
        month=1,
        bucket="test-bucket",
        ingestion_result={
            "status": "REUSED",
            "metadata": {
                "size": len(raw_bytes),
                "checksum_sha256": ("wrong-checksum" if bad_checksum else checksum),
                "version_id": "raw-v1",
                "encryption": "AES256",
            },
        },
    )

    class FakeS3:
        def get_object(self, **kwargs):
            if kwargs["Key"] == reference["manifest_key"]:
                return {"Body": BytesIO(json.dumps(manifest).encode())}

            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == reference["raw_key"]
            assert kwargs["VersionId"] == "raw-v1"

            return {
                "Body": BytesIO(raw_bytes),
                "VersionId": "raw-v1",
            }

    if bad_checksum:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            download_versioned_raw(FakeS3(), reference, tmp_path)

        assert list(tmp_path.iterdir()) == []

    else:
        local_path = download_versioned_raw(FakeS3(), reference, tmp_path)

        assert Path(local_path).read_bytes() == raw_bytes
        assert Path(local_path).suffix == ".parquet"


def test_resolve_versioned_raw_paths(tmp_path, monkeypatch):
    calls = []

    references = [
        {"data_month": "2022-12"},
        {"data_month": "2023-01"},
    ]

    def fake_download(s3_client, reference, output_dir):
        calls.append(reference["data_month"])

        return str(tmp_path / f"yellow_tripdata_{reference['data_month']}.parquet")

    monkeypatch.setattr(
        s3_raw_reader,
        "download_versioned_raw",
        fake_download,
    )

    paths = s3_raw_reader.resolve_versioned_raw_paths(
        s3_client=object(),
        raw_file_months=["2022-12", "2023-01"],
        raw_references=references,
        output_dir=tmp_path,
    )

    assert calls == ["2022-12", "2023-01"]
    assert len(paths) == 2
    assert paths[0].endswith("2022-12.parquet")
    assert paths[1].endswith("2023-01.parquet")


def test_resolve_rejects_wrong_months(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/should-not-exist.parquet"

    monkeypatch.setattr(
        s3_raw_reader,
        "download_versioned_raw",
        fake_download,
    )

    references = [
        {"data_month": "2024-01"},
        {"data_month": "2024-03"},
    ]

    with pytest.raises(
        ValueError,
        match="do not match execution read scope",
    ):
        s3_raw_reader.resolve_versioned_raw_paths(
            s3_client=object(),
            raw_file_months=["2024-01", "2024-02"],
            raw_references=references,
            output_dir=tmp_path,
        )

    assert calls == []
