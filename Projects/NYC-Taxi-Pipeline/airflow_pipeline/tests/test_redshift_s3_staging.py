"""R1/R2 release-boundary tests; no real AWS calls or Spark required."""

import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from scripts.redshift_s3_staging import (
    UploadItem,
    build_inventory,
    put_verified,
)
from scripts.serving_snapshot_integrity import (
    TABLES,
    digest_file,
    publish_seal,
    verify_seal,
)


@pytest.fixture
def export(tmp_path):
    release = "airflow-demo-001"
    root = tmp_path / "serving_exports"
    folder = root / release
    folder.mkdir(parents=True)
    info = {}
    for name in TABLES:
        table_dir = folder / name
        table_dir.mkdir()
        number = 2 if name == "od_corridor_daily" else 1
        for i in range(number):
            (table_dir / f"part-{i:05d}-abcd.parquet").write_bytes(
                f"PAR1:{name}:{i}".encode()
            )
        info[name] = {
            "parquet_path": str(table_dir),
            "row_count": 3,
            "columns": ["value"],
            "staging_table": f"{name}_r123",
            "create_table_sql": f'CREATE TABLE "{name}" ("value" BIGINT);',
        }
    plan = {
        "release_id": release,
        "source_versions": {name: 0 for name in TABLES},
        "logical_window": {"start": "2023-01-01", "end_exclusive": "2023-04-01"},
        "tables": info,
    }
    (folder / "export_plan.json").write_text(json.dumps(plan))
    return root, release, folder


def test_inventory_requires_r1_seal(export):
    root, release, _ = export
    with pytest.raises(ValueError, match="inventory missing"):
        build_inventory(root, release, "nyc-test-bucket")


def test_sealed_inventory_seven_files_fourteen_objects(export):
    root, release, folder = export
    publish_seal(folder)
    result = build_inventory(root, release, "nyc-test-bucket")
    assert result["parquet_count"] == 7
    assert len(result["objects"]) == 14
    assert result["objects"][-1].key.endswith("release_manifest.json")
    manifests = [x for x in result["objects"] if "/copy_manifests/" in x.key]
    assert len(manifests) == 6
    corr = [x for x in manifests if x.key.endswith("/od_corridor_daily.json")][0]
    entries = json.loads(corr.body)["entries"]
    assert len(entries) == 2
    assert all(x["mandatory"] and x["meta"]["content_length"] > 0 for x in entries)


def test_one_of_two_parquet_files_missing_is_detected(export):
    root, release, folder = export
    publish_seal(folder)
    (folder / "od_corridor_daily" / "part-00001-abcd.parquet").unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        build_inventory(root, release, "nyc-test-bucket")


def test_same_size_file_mutation_is_detected(export):
    root, release, folder = export
    publish_seal(folder)
    target = folder / "trip_summary_hourly" / "part-00000-abcd.parquet"
    target.write_bytes(b"X" * target.stat().st_size)
    with pytest.raises(ValueError, match="inventory mismatch"):
        build_inventory(root, release, "nyc-test-bucket")


def test_extra_parquet_is_detected(export):
    root, release, folder = export
    publish_seal(folder)
    (folder / "trip_summary_hourly" / "part-added.parquet").write_bytes(b"PAR1")
    with pytest.raises(ValueError, match="inventory mismatch"):
        build_inventory(root, release, "nyc-test-bucket")


def test_modified_plan_is_detected(export):
    root, release, folder = export
    publish_seal(folder)
    plan_path = folder / "export_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["tables"][TABLES[0]]["row_count"] += 1
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="header mismatch"):
        build_inventory(root, release, "nyc-test-bucket")


def test_wrong_path_rejected_before_seal(export):
    _, _, folder = export
    path = folder / "export_plan.json"
    plan = json.loads(path.read_text())
    plan["tables"][TABLES[0]]["parquet_path"] = "/tmp"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Unexpected R1 export directory"):
        publish_seal(folder)


def test_seal_refuses_to_overwrite_changed_evidence(export):
    _, _, folder = export
    publish_seal(folder)
    (folder / "od_corridor_daily" / "part-00001-abcd.parquet").unlink()
    with pytest.raises(ValueError, match="differs"):
        publish_seal(folder)


def test_digester_streams_large_file(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abc" * 2_000_000)
    size, digest = digest_file(path)
    assert size == 6_000_000
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


class FakeS3:
    def __init__(self):
        self.data = {}
        self.metadata = {}
        self.pending = {}
        self.events = []
        self.aborted = 0

    def _identity(self, kwargs):
        return (kwargs["Bucket"], kwargs["Key"])

    def _exists(self):
        return ClientError(
            {"Error": {"Code": "PreconditionFailed", "Message": "Already exists"}},
            "PutObject",
        )

    def put_object(self, **kwargs):
        identity = self._identity(kwargs)
        assert kwargs["IfNoneMatch"] == "*"
        if identity in self.data:
            raise self._exists()
        contents = kwargs["Body"].read()
        assert kwargs["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(contents).digest()
        ).decode()
        self.data[identity] = contents
        self.metadata[identity] = kwargs["Metadata"]
        self.events.append(("put", identity[1]))

    def head_object(self, **kwargs):
        identity = self._identity(kwargs)
        return {
            "ContentLength": len(self.data[identity]),
            "Metadata": self.metadata[identity],
        }

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(self.data[self._identity(kwargs)])}

    def create_multipart_upload(self, **kwargs):
        identity = self._identity(kwargs)
        assert kwargs["ChecksumAlgorithm"] == "SHA256"
        upload_id = str(len(self.pending) + 1)
        self.pending[upload_id] = (identity, kwargs["Metadata"], {})
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs):
        upload_id = kwargs["UploadId"]
        data = kwargs["Body"]
        assert kwargs["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(data).digest()
        ).decode()
        self.pending[upload_id][2][kwargs["PartNumber"]] = data
        return {"ETag": f'"part-{kwargs["PartNumber"]}"'}

    def complete_multipart_upload(self, **kwargs):
        identity, meta, parts = self.pending[kwargs["UploadId"]]
        assert kwargs["IfNoneMatch"] == "*"
        if identity in self.data:
            raise self._exists()
        self.data[identity] = b"".join(parts[number] for number in sorted(parts))
        self.metadata[identity] = meta
        self.events.append(("complete", identity[1]))

    def abort_multipart_upload(self, **kwargs):
        self.aborted += 1
        del self.pending[kwargs["UploadId"]]


def _byte_item(key, body):
    return UploadItem(
        key=key,
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        content_type="application/octet-stream",
        body=body,
    )


def test_small_upload_is_create_only_and_safe_to_retry():
    s3 = FakeS3()
    item = _byte_item("serving/f", b"hello")
    assert put_verified(s3, "bucket", item) == "CREATED"
    assert put_verified(s3, "bucket", item) == "VERIFIED_EXISTING"
    assert s3.data[("bucket", item.key)] == b"hello"


def test_same_s3_key_with_different_data_fails_closed():
    s3 = FakeS3()
    put_verified(s3, "bucket", _byte_item("serving/f", b"first"))
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        put_verified(s3, "bucket", _byte_item("serving/f", b"other"))


def test_multipart_splits_data_and_detects_existing(monkeypatch):
    import scripts.redshift_s3_staging as module
    monkeypatch.setattr(module, "MULTIPART_THRESHOLD", 5)
    monkeypatch.setattr(module, "PART_SIZE", 6)
    s3 = FakeS3()
    item = _byte_item("serving/large", b"12345678901234567890")
    assert put_verified(s3, "bucket", item) == "CREATED"
    assert s3.data[("bucket", item.key)] == item.body
    assert put_verified(s3, "bucket", item) == "VERIFIED_EXISTING"
    assert s3.aborted == 1


def test_upload_manifest_is_last(export):
    root, release, folder = export
    publish_seal(folder)
    plan = build_inventory(root, release, "nyc-test-bucket")
    s3 = FakeS3()
    for item in plan["objects"]:
        assert put_verified(s3, plan["bucket"], item) == "CREATED"
    assert s3.events[-1][1].endswith("release_manifest.json")
    assert len(s3.data) == 14
    for item in plan["objects"]:
        assert put_verified(s3, plan["bucket"], item) == "VERIFIED_EXISTING"


def test_local_file_changed_after_preflight_fails_upload(export):
    root, release, folder = export
    publish_seal(folder)
    plan = build_inventory(root, release, "nyc-test-bucket")
    item = plan["objects"][0]
    item.source_path.write_bytes(b"X" * item.size_bytes)
    with pytest.raises(AssertionError):  # FakeS3 simulates S3 rejecting the SHA-256
        put_verified(FakeS3(), "nyc-test-bucket", item)


def test_release_not_published_if_a_data_upload_fails(export):
    root, release, folder = export
    publish_seal(folder)
    result = build_inventory(root, release, "nyc-test-bucket")
    s3 = FakeS3()
    first = result["objects"][0]
    put_verified(s3, result["bucket"], first)
    second = result["objects"][1]
    s3.data[(result["bucket"], second.key)] = b"not the expected data"
    s3.metadata[(result["bucket"], second.key)] = {"sha256": "incorrect"}
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        put_verified(s3, result["bucket"], second)
    assert (result["bucket"], result["release_manifest_key"]) not in s3.data


def test_remote_modified_after_upload_is_rejected_on_retry():
    s3 = FakeS3()
    item = _byte_item("serving/object", b"original contents")
    put_verified(s3, "bucket", item)
    s3.data[("bucket", item.key)] = b"CHANGED! contents"
    # A forged metadata header is not sufficient to make this retry pass.
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        put_verified(s3, "bucket", item)
