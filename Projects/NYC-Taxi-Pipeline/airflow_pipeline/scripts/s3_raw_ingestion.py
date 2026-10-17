"""NYC V2 — AWS S3 raw ingestion."""

import boto3
import json
import base64
import hashlib
from pathlib import Path
from botocore.client import BaseClient
from botocore.exceptions import ClientError


def create_s3_client(
    region: str = "eu-central-1",
    profile: str | None = None,
) -> BaseClient:
    """Create an S3 client using the configured AWS identity."""

    session = boto3.Session(
        profile_name=profile,
        region_name=region,
    )

    return session.client("s3")


def build_raw_object_key(year: int, month: int) -> str:
    """Build the canonical S3 key for NYC TLC Yellow Taxi raw data."""

    if not 1 <= year <= 9999:
        raise ValueError(f"Invalid year: {year}")

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {month}")

    filename = f"yellow_tripdata_{year:04d}-{month:02d}.parquet"

    return f"raw/nyc_tlc/yellow/" f"{year:04d}/{month:02d}/{filename}"


def build_raw_manifest_key(year: int, month: int) -> str:
    """Build the S3 manifest key for a monthly raw object."""

    raw_key = build_raw_object_key(year, month)

    return f"{raw_key}.manifest.json"


def get_raw_object_metadata(
    s3_client: BaseClient,
    bucket: str,
    key: str,
) -> dict | None:
    """Read S3 object metadata, returning None only for a missing object."""

    try:
        response = s3_client.head_object(
            Bucket=bucket,
            Key=key,
            ChecksumMode="ENABLED",
        )

        return {
            "size": response["ContentLength"],
            "checksum_sha256": response.get("ChecksumSHA256"),
            "version_id": response.get("VersionId"),
            "encryption": response.get("ServerSideEncryption"),
        }

    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))

        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None

        raise


def verify_existing_raw_object(
    s3_client: BaseClient,
    bucket: str,
    key: str,
    local_path: str | Path,
) -> dict:
    """Verify an existing S3 object's integrity against a local file."""

    path = Path(local_path)

    metadata = get_raw_object_metadata(
        s3_client,
        bucket,
        key,
    )

    if metadata is None:
        raise FileNotFoundError(f"S3 object does not exist: {key}")

    remote_checksum = metadata["checksum_sha256"]

    if not remote_checksum:
        raise ValueError("S3 object is missing SHA-256 checksum")

    local_size = path.stat().st_size

    if local_size != metadata["size"]:
        raise ValueError("S3 object size mismatch")

    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").digest()

    local_checksum = base64.b64encode(digest).decode("ascii")

    if local_checksum != remote_checksum:
        raise ValueError("S3 object SHA-256 mismatch")

    return metadata


def upload_new_raw_object(
    s3_client: BaseClient,
    bucket: str,
    key: str,
    local_path: str | Path,
) -> dict:
    """Upload a new S3 raw object without overwriting an existing key."""

    path = Path(local_path)

    if path.stat().st_size == 0:
        raise ValueError("Cannot upload an empty raw file")

    with path.open("rb") as file:
        digest = hashlib.file_digest(
            file,
            "sha256",
        ).digest()

    checksum = base64.b64encode(digest).decode("ascii")

    with path.open("rb") as file:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file,
            IfNoneMatch="*",
            ChecksumSHA256=checksum,
            ServerSideEncryption="AES256",
        )

    return response


def ensure_raw_object(
    s3_client: BaseClient,
    bucket: str,
    key: str,
    local_path: str | Path,
) -> dict:
    """Upload or safely reuse a verified S3 raw object."""

    metadata = get_raw_object_metadata(s3_client, bucket, key)

    if metadata is not None:
        verified = verify_existing_raw_object(s3_client, bucket, key, local_path)

        return {
            "status": "REUSED",
            "metadata": verified,
        }

    try:
        response = upload_new_raw_object(s3_client, bucket, key, local_path)

    except ClientError as exc:
        error = exc.response.get("Error", {})
        http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if http_status != 412 and error.get("Code") not in {
            "PreconditionFailed",
            "412",
        }:
            raise

        # Another writer may have created the object.
        verified = verify_existing_raw_object(s3_client, bucket, key, local_path)

        return {
            "status": "REUSED_AFTER_RACE",
            "metadata": verified,
        }

    verified = verify_existing_raw_object(s3_client, bucket, key, local_path)

    uploaded_version = response.get("VersionId")

    if uploaded_version is not None and verified["version_id"] != uploaded_version:
        raise RuntimeError("S3 object version changed after upload")

    return {
        "status": "UPLOADED",
        "metadata": verified,
    }


def build_raw_manifest(
    year: int,
    month: int,
    bucket: str,
    ingestion_result: dict,
) -> dict:
    """Build a manifest from a successfully verified S3 raw object."""

    valid_statuses = {
        "UPLOADED",
        "REUSED",
        "REUSED_AFTER_RACE",
    }

    if ingestion_result.get("status") not in valid_statuses:
        raise ValueError("Raw object has not been verified")

    metadata = ingestion_result["metadata"]

    required_fields = {
        "size",
        "checksum_sha256",
        "version_id",
    }

    if any(not metadata.get(field) for field in required_fields):
        raise ValueError("Incomplete S3 object metadata")

    if metadata.get("encryption") != "AES256":
        raise ValueError("Unexpected S3 encryption")

    raw_key = build_raw_object_key(year, month)

    return {
        "schema_version": 1,
        "source": "NYC TLC Yellow Taxi",
        "source_url": (
            "https://d37ci6vzurychx.cloudfront.net/"
            f"trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
        ),
        "data_month": f"{year:04d}-{month:02d}",
        "s3": {
            "bucket": bucket,
            "key": raw_key,
            "version_id": metadata["version_id"],
            "size_bytes": metadata["size"],
            "checksum_sha256": metadata["checksum_sha256"],
            "encryption": metadata["encryption"],
        },
        "verification": {
            "scope": "local_vs_s3",
            "result": "PASS",
        },
    }


def publish_raw_manifest(
    s3_client: BaseClient,
    bucket: str,
    manifest_key: str,
    manifest: dict,
) -> dict:
    """Conditionally publish an immutable S3 raw manifest."""

    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")

    try:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=payload,
            IfNoneMatch="*",
            ContentType="application/json",
            ChecksumSHA256=checksum,
            ServerSideEncryption="AES256",
        )

        return {
            "status": "PUBLISHED",
            "manifest_key": manifest_key,
            "version_id": response.get("VersionId"),
        }

    except ClientError as exc:
        error = exc.response.get("Error", {})
        http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if http_status != 412 and error.get("Code") not in {
            "PreconditionFailed",
            "412",
        }:
            raise

    # A manifest already exists. Read and compare it.
    response = s3_client.get_object(
        Bucket=bucket,
        Key=manifest_key,
    )

    body = response["Body"]

    try:
        existing_bytes = body.read(65537)
    finally:
        body.close()

    if len(existing_bytes) > 65536:
        raise ValueError("Existing manifest exceeds size limit")

    try:
        existing_manifest = json.loads(existing_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Existing manifest is invalid JSON") from exc

    if existing_manifest != manifest:
        raise ValueError("S3 manifest conflict")

    return {
        "status": "REUSED",
        "manifest_key": manifest_key,
        "version_id": response.get("VersionId"),
    }


def ingest_month(
    s3_client: BaseClient,
    bucket: str,
    year: int,
    month: int,
    local_path: str | Path,
) -> dict:
    """Ingest one validated monthly Parquet into S3 with a manifest."""

    raw_key = build_raw_object_key(year, month)
    manifest_key = build_raw_manifest_key(year, month)

    # Step 1: Upload or safely reuse verified raw data.
    raw_result = ensure_raw_object(
        s3_client=s3_client,
        bucket=bucket,
        key=raw_key,
        local_path=local_path,
    )

    # Step 2: Build evidence for the verified object.
    manifest = build_raw_manifest(
        year=year,
        month=month,
        bucket=bucket,
        ingestion_result=raw_result,
    )

    # Step 3: Publish or safely reuse the manifest.
    # Publication failure propagates to the caller.
    manifest_result = publish_raw_manifest(
        s3_client=s3_client,
        bucket=bucket,
        manifest_key=manifest_key,
        manifest=manifest,
    )

    return {
        "data_month": f"{year:04d}-{month:02d}",
        "raw_status": raw_result["status"],
        "raw_key": raw_key,
        "raw_version_id": raw_result["metadata"]["version_id"],
        "manifest_status": manifest_result["status"],
        "manifest_key": manifest_key,
    }
