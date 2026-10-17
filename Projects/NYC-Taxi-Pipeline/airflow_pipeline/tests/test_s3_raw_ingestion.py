import json
import pytest
import base64
import hashlib
from io import BytesIO
from scripts import s3_raw_ingestion
from botocore.exceptions import ClientError


def test_create_s3_client(monkeypatch):
    calls = {}

    class FakeSession:
        def __init__(self, profile_name, region_name):
            calls["profile"] = profile_name
            calls["region"] = region_name

        def client(self, service_name):
            calls["service"] = service_name
            return "fake-s3-client"

    monkeypatch.setattr(
        s3_raw_ingestion.boto3,
        "Session",
        FakeSession,
    )

    client = s3_raw_ingestion.create_s3_client(
        region="eu-central-1",
        profile="nyc-v2-dev",
    )

    assert client == "fake-s3-client"
    assert calls == {
        "profile": "nyc-v2-dev",
        "region": "eu-central-1",
        "service": "s3",
    }


def test_build_raw_object_key():
    key = s3_raw_ingestion.build_raw_object_key(
        year=2024,
        month=1,
    )

    assert key == ("raw/nyc_tlc/yellow/2024/01/" "yellow_tripdata_2024-01.parquet")


def test_build_raw_object_key_rejects_invalid_month():
    import pytest

    with pytest.raises(ValueError, match="Invalid month"):
        s3_raw_ingestion.build_raw_object_key(
            year=2024,
            month=13,
        )


def test_existing_s3_object_returns_metadata():
    class FakeS3:
        def head_object(self, **kwargs):
            assert kwargs["ChecksumMode"] == "ENABLED"

            return {
                "ContentLength": 49961641,
                "ChecksumSHA256": "example-checksum",
                "VersionId": "version-001",
                "ServerSideEncryption": "AES256",
            }

    result = s3_raw_ingestion.get_raw_object_metadata(
        FakeS3(),
        "test-bucket",
        "raw/test.parquet",
    )

    assert result["size"] == 49961641
    assert result["version_id"] == "version-001"
    assert result["encryption"] == "AES256"


def test_missing_s3_object_returns_none():
    class FakeS3:
        def head_object(self, **kwargs):
            raise ClientError(
                {
                    "Error": {
                        "Code": "404",
                        "Message": "Not Found",
                    }
                },
                "HeadObject",
            )

    result = s3_raw_ingestion.get_raw_object_metadata(
        FakeS3(),
        "test-bucket",
        "raw/missing.parquet",
    )

    assert result is None


def test_s3_access_denied_is_not_treated_as_missing():
    class FakeS3:
        def head_object(self, **kwargs):
            raise ClientError(
                {
                    "Error": {
                        "Code": "403",
                        "Message": "Access Denied",
                    }
                },
                "HeadObject",
            )

    with pytest.raises(ClientError):
        s3_raw_ingestion.get_raw_object_metadata(
            FakeS3(),
            "test-bucket",
            "raw/test.parquet",
        )


def test_existing_s3_object_checksum_matches(tmp_path):
    data = b"example parquet bytes"
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(data)

    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")

    class FakeS3:
        def head_object(self, **kwargs):
            return {
                "ContentLength": len(data),
                "ChecksumSHA256": checksum,
                "VersionId": "version-001",
                "ServerSideEncryption": "AES256",
            }

    result = s3_raw_ingestion.verify_existing_raw_object(
        FakeS3(),
        "test-bucket",
        "raw/sample.parquet",
        local_file,
    )

    assert result["size"] == len(data)
    assert result["checksum_sha256"] == checksum
    assert result["version_id"] == "version-001"


@pytest.mark.parametrize(
    "scenario, expected_error",
    [
        ("wrong_size", "size mismatch"),
        ("wrong_checksum", "SHA-256 mismatch"),
        ("missing_checksum", "missing SHA-256"),
    ],
)
def test_existing_s3_object_integrity_failure(
    tmp_path,
    scenario,
    expected_error,
):
    data = b"example parquet bytes"
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(data)

    checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")

    class FakeS3:
        def head_object(self, **kwargs):
            return {
                "ContentLength": (
                    len(data) + 1 if scenario == "wrong_size" else len(data)
                ),
                "ChecksumSHA256": (
                    None
                    if scenario == "missing_checksum"
                    else (
                        "incorrect-checksum"
                        if scenario == "wrong_checksum"
                        else checksum
                    )
                ),
                "VersionId": "version-001",
            }

    with pytest.raises(ValueError, match=expected_error):
        s3_raw_ingestion.verify_existing_raw_object(
            FakeS3(),
            "test-bucket",
            "raw/sample.parquet",
            local_file,
        )


def test_conditional_upload_succeeds(tmp_path):
    data = b"sample raw data"
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(data)

    calls = {}

    class FakeS3:
        def put_object(self, **kwargs):
            calls.update(kwargs)
            calls["uploaded_bytes"] = kwargs["Body"].read()

            return {
                "ETag": '"example-etag"',
                "VersionId": "version-001",
            }

    result = s3_raw_ingestion.upload_new_raw_object(
        FakeS3(),
        "test-bucket",
        "raw/sample.parquet",
        local_file,
    )

    expected_checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")

    assert calls["Bucket"] == "test-bucket"
    assert calls["Key"] == "raw/sample.parquet"
    assert calls["IfNoneMatch"] == "*"
    assert calls["ChecksumSHA256"] == expected_checksum
    assert calls["ServerSideEncryption"] == "AES256"
    assert calls["uploaded_bytes"] == data
    assert result["VersionId"] == "version-001"


def test_conditional_upload_rejects_existing_object(tmp_path):
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(b"sample raw data")

    class FakeS3:
        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*"

            raise ClientError(
                {
                    "Error": {
                        "Code": "PreconditionFailed",
                        "Message": "Object already exists",
                    },
                    "ResponseMetadata": {
                        "HTTPStatusCode": 412,
                    },
                },
                "PutObject",
            )

    with pytest.raises(ClientError) as exc_info:
        s3_raw_ingestion.upload_new_raw_object(
            FakeS3(),
            "test-bucket",
            "raw/sample.parquet",
            local_file,
        )

    assert exc_info.value.response["Error"]["Code"] == "PreconditionFailed"


@pytest.mark.parametrize(
    "already_exists",
    [True, False],
)
def test_ensure_raw_object_is_idempotent(
    tmp_path,
    monkeypatch,
    already_exists,
):
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(b"example-data")

    calls = {"upload": 0, "verify": 0}

    metadata = {
        "size": 12,
        "checksum_sha256": "verified-checksum",
        "version_id": "v1",
    }

    def fake_get_metadata(*args):
        return metadata if already_exists else None

    def fake_verify(*args):
        calls["verify"] += 1
        return metadata

    def fake_upload(*args):
        calls["upload"] += 1
        return {"VersionId": "v1"}

    monkeypatch.setattr(
        s3_raw_ingestion,
        "get_raw_object_metadata",
        fake_get_metadata,
    )

    monkeypatch.setattr(
        s3_raw_ingestion,
        "verify_existing_raw_object",
        fake_verify,
    )

    monkeypatch.setattr(
        s3_raw_ingestion,
        "upload_new_raw_object",
        fake_upload,
    )

    result = s3_raw_ingestion.ensure_raw_object(
        object(),
        "test-bucket",
        "raw/sample.parquet",
        local_file,
    )

    assert result["status"] == ("REUSED" if already_exists else "UPLOADED")

    assert calls["upload"] == (0 if already_exists else 1)

    assert calls["verify"] == 1


@pytest.mark.parametrize("content_matches", [True, False])
def test_ensure_raw_object_handles_upload_race(
    tmp_path,
    monkeypatch,
    content_matches,
):
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(b"example-data")

    calls = {"upload": 0, "verify": 0}

    def fake_get_metadata(*args):
        # Object did not exist during the initial HEAD.
        return None

    def fake_upload(*args):
        calls["upload"] += 1

        # Another worker created the object first.
        raise ClientError(
            {
                "Error": {
                    "Code": "PreconditionFailed",
                    "Message": "Object already exists",
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": 412,
                },
            },
            "PutObject",
        )

    def fake_verify(*args):
        calls["verify"] += 1

        if not content_matches:
            raise ValueError("S3 object SHA-256 mismatch")

        return {
            "size": 12,
            "checksum_sha256": "verified-checksum",
            "version_id": "other-worker-version",
        }

    monkeypatch.setattr(
        s3_raw_ingestion,
        "get_raw_object_metadata",
        fake_get_metadata,
    )
    monkeypatch.setattr(
        s3_raw_ingestion,
        "upload_new_raw_object",
        fake_upload,
    )
    monkeypatch.setattr(
        s3_raw_ingestion,
        "verify_existing_raw_object",
        fake_verify,
    )

    def run_ingestion():
        return s3_raw_ingestion.ensure_raw_object(
            object(),
            "test-bucket",
            "raw/sample.parquet",
            local_file,
        )

    if content_matches:
        result = run_ingestion()
        assert result["status"] == "REUSED_AFTER_RACE"
    else:
        with pytest.raises(
            ValueError,
            match="SHA-256 mismatch",
        ):
            run_ingestion()

    assert calls["upload"] == 1
    assert calls["verify"] == 1


def test_ensure_raw_object_propagates_access_denied(
    tmp_path,
    monkeypatch,
):
    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(b"example-data")

    monkeypatch.setattr(
        s3_raw_ingestion,
        "get_raw_object_metadata",
        lambda *args: None,
    )

    def fake_upload(*args):
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "Access Denied",
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": 403,
                },
            },
            "PutObject",
        )

    monkeypatch.setattr(
        s3_raw_ingestion,
        "upload_new_raw_object",
        fake_upload,
    )

    with pytest.raises(ClientError) as exc_info:
        s3_raw_ingestion.ensure_raw_object(
            object(),
            "test-bucket",
            "raw/sample.parquet",
            local_file,
        )

    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"


def test_build_raw_manifest_key():
    key = s3_raw_ingestion.build_raw_manifest_key(
        year=2024,
        month=1,
    )

    assert key == (
        "raw/nyc_tlc/yellow/2024/01/" "yellow_tripdata_2024-01.parquet.manifest.json"
    )


def test_build_raw_manifest_key_rejects_invalid_month():
    with pytest.raises(ValueError, match="Invalid month"):
        s3_raw_ingestion.build_raw_manifest_key(
            year=2024,
            month=13,
        )


def test_build_raw_manifest():
    import json

    result = {
        "status": "UPLOADED",
        "metadata": {
            "size": 49961641,
            "checksum_sha256": "example-checksum",
            "version_id": "version-001",
            "encryption": "AES256",
        },
    }

    manifest = s3_raw_ingestion.build_raw_manifest(
        year=2024,
        month=1,
        bucket="test-bucket",
        ingestion_result=result,
    )

    assert manifest["schema_version"] == 1
    assert manifest["data_month"] == "2024-01"
    assert manifest["s3"]["size_bytes"] == 49961641
    assert manifest["s3"]["version_id"] == "version-001"
    assert manifest["verification"]["result"] == "PASS"

    # The manifest must be JSON serializable.
    serialized = json.dumps(manifest, sort_keys=True)
    assert json.loads(serialized) == manifest


def test_build_raw_manifest_rejects_unverified_object():
    result = {
        "status": "FAILED",
        "metadata": {},
    }

    with pytest.raises(ValueError, match="not been verified"):
        s3_raw_ingestion.build_raw_manifest(
            year=2024,
            month=1,
            bucket="test-bucket",
            ingestion_result=result,
        )


def test_manifest_publication_succeeds():
    calls = {}

    class FakeS3:
        def put_object(self, **kwargs):
            calls.update(kwargs)
            return {"VersionId": "manifest-v1"}

    manifest = {"schema_version": 1, "data_month": "2024-01"}

    result = s3_raw_ingestion.publish_raw_manifest(
        FakeS3(), "bucket", "raw/test.manifest.json", manifest
    )

    assert result["status"] == "PUBLISHED"
    assert result["version_id"] == "manifest-v1"
    assert calls["IfNoneMatch"] == "*"
    assert calls["ContentType"] == "application/json"
    assert calls["ServerSideEncryption"] == "AES256"
    assert json.loads(calls["Body"]) == manifest


def test_existing_identical_manifest_is_reused():
    manifest = {"schema_version": 1, "data_month": "2024-01"}

    class FakeS3:
        def put_object(self, **kwargs):
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )

        def get_object(self, **kwargs):
            return {
                "Body": BytesIO(json.dumps(manifest).encode()),
                "VersionId": "existing-v1",
            }

    result = s3_raw_ingestion.publish_raw_manifest(
        FakeS3(), "bucket", "raw/test.manifest.json", manifest
    )

    assert result["status"] == "REUSED"
    assert result["version_id"] == "existing-v1"


def test_existing_conflicting_manifest_is_rejected():
    manifest = {"schema_version": 1, "data_month": "2024-01"}

    class FakeS3:
        def put_object(self, **kwargs):
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )

        def get_object(self, **kwargs):
            return {"Body": BytesIO(b'{"schema_version":1,"data_month":"2024-02"}')}

    with pytest.raises(ValueError, match="manifest conflict"):
        s3_raw_ingestion.publish_raw_manifest(
            FakeS3(), "bucket", "raw/test.manifest.json", manifest
        )


def test_manifest_access_denied_propagates():
    class FakeS3:
        def put_object(self, **kwargs):
            raise ClientError(
                {
                    "Error": {"Code": "AccessDenied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "PutObject",
            )

    with pytest.raises(ClientError):
        s3_raw_ingestion.publish_raw_manifest(
            FakeS3(),
            "bucket",
            "raw/test.manifest.json",
            {"schema_version": 1},
        )


def test_ingest_month_success(tmp_path, monkeypatch):
    local_file = tmp_path / "yellow_tripdata_2024-01.parquet"
    local_file.write_bytes(b"sample-data")

    calls = []

    metadata = {
        "size": 11,
        "checksum_sha256": "example-checksum",
        "version_id": "raw-v1",
        "encryption": "AES256",
    }

    def fake_ensure(**kwargs):
        calls.append("ensure")
        return {"status": "UPLOADED", "metadata": metadata}

    def fake_publish(**kwargs):
        calls.append("publish")
        assert kwargs["manifest"]["s3"]["version_id"] == "raw-v1"
        return {"status": "PUBLISHED", "version_id": "manifest-v1"}

    monkeypatch.setattr(s3_raw_ingestion, "ensure_raw_object", fake_ensure)
    monkeypatch.setattr(s3_raw_ingestion, "publish_raw_manifest", fake_publish)

    result = s3_raw_ingestion.ingest_month(object(), "test-bucket", 2024, 1, local_file)

    assert calls == ["ensure", "publish"]
    assert result["raw_status"] == "UPLOADED"
    assert result["manifest_status"] == "PUBLISHED"
    assert result["raw_version_id"] == "raw-v1"


def test_ingest_month_recovers_after_manifest_failure(tmp_path, monkeypatch):
    local_file = tmp_path / "yellow_tripdata_2024-01.parquet"
    local_file.write_bytes(b"sample-data")

    calls = {"upload": 0, "publish": 0}
    raw_exists = False

    metadata = {
        "size": 11,
        "checksum_sha256": "example-checksum",
        "version_id": "raw-v1",
        "encryption": "AES256",
    }

    def fake_ensure(**kwargs):
        nonlocal raw_exists

        if raw_exists:
            return {"status": "REUSED", "metadata": metadata}

        calls["upload"] += 1
        raw_exists = True
        return {"status": "UPLOADED", "metadata": metadata}

    def fake_publish(**kwargs):
        calls["publish"] += 1

        if calls["publish"] == 1:
            raise RuntimeError("Manifest publication interrupted")

        return {"status": "PUBLISHED", "version_id": "manifest-v1"}

    monkeypatch.setattr(s3_raw_ingestion, "ensure_raw_object", fake_ensure)
    monkeypatch.setattr(s3_raw_ingestion, "publish_raw_manifest", fake_publish)

    with pytest.raises(RuntimeError, match="Manifest publication interrupted"):
        s3_raw_ingestion.ingest_month(object(), "test-bucket", 2024, 1, local_file)

    # Retry the same month.
    result = s3_raw_ingestion.ingest_month(object(), "test-bucket", 2024, 1, local_file)

    assert calls["upload"] == 1
    assert calls["publish"] == 2
    assert result["raw_status"] == "REUSED"
    assert result["manifest_status"] == "PUBLISHED"
