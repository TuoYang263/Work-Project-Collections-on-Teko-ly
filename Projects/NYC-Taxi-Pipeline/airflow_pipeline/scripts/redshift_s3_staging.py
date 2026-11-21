"""NYC V2 R2: stream verified, immutable R1 Parquet snapshots to S3.

Default mode only reads local files. --upload makes AWS writes.
No Redshift deployment or SQL execution is performed here.
"""

import argparse
import base64
import hashlib
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from scripts.serving_snapshot_integrity import (
    BUFFER_SIZE,
    RELEASE_PATTERN,
    TABLES,
    digest_file,
    json_bytes,
    verify_seal,
)

PREFIX = "serving/releases"
BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{2,62}\Z")
MULTIPART_THRESHOLD = 64 * 1024 * 1024
PART_SIZE = 64 * 1024 * 1024
MAX_MULTIPART_PARTS = 10_000


@dataclass(frozen=True)
class UploadItem:
    key: str
    size_bytes: int
    sha256: str
    content_type: str
    source_path: Path | None = None
    body: bytes | None = None

    def open(self):
        if self.source_path is not None:
            return self.source_path.open("rb")
        if self.body is None:
            raise ValueError(f"Object has no body: {self.key}")
        return io.BytesIO(self.body)


def _from_json(key: str, data: dict) -> UploadItem:
    payload = json_bytes(data)
    return UploadItem(
        key=key,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        content_type="application/json",
        body=payload,
    )


def build_inventory(output_root: Path, release_id: str, bucket: str) -> dict:
    """Fail closed on exact file inventory, size, and SHA-256 differences."""
    if not isinstance(release_id, str) or not RELEASE_PATTERN.fullmatch(release_id):
        raise ValueError("Unsafe release ID")
    if not isinstance(bucket, str) or not BUCKET_PATTERN.fullmatch(bucket):
        raise ValueError("Invalid S3 bucket")
    root = Path(output_root).resolve(strict=True)
    release_dir = (root / release_id).resolve(strict=True)
    if release_dir.parent != root:
        raise ValueError("Release directory escaped export root")
    plan, seal = verify_seal(release_dir)
    objects = []
    manifest = {
        "schema_version": 2,
        "gold_release_id": release_id,
        "logical_window": plan["logical_window"],
        "gold_delta_versions": plan["source_versions"],
        "r1_inventory_sha256": hashlib.sha256(
            (release_dir / "file_inventory.json").read_bytes()
        ).hexdigest(),
        "tables": {},
    }
    prefix = f"{PREFIX}/{release_id}"
    total_files = 0
    total_bytes = 0
    for name in TABLES:
        info = plan["tables"][name]
        entries = []
        file_evidence = []
        for file_info in seal["tables"][name]:
            path = release_dir / name / file_info["name"]
            size = file_info["size_bytes"]
            sha = file_info["sha256"]
            key = f"{prefix}/{name}/{path.name}"
            uri = f"s3://{bucket}/{key}"
            objects.append(
                UploadItem(
                    key=key,
                    source_path=path,
                    size_bytes=size,
                    sha256=sha,
                    content_type="application/octet-stream",
                )
            )
            entries.append(
                {"url": uri, "mandatory": True, "meta": {"content_length": size}}
            )
            file_evidence.append({"url": uri, "size_bytes": size, "sha256": sha})
            total_files += 1
            total_bytes += size
        manifest_key = f"{prefix}/copy_manifests/{name}.json"
        objects.append(_from_json(manifest_key, {"entries": entries}))
        if (
            not isinstance(info.get("columns"), list)
            or not info["columns"]
            or not isinstance(info.get("create_table_sql"), str)
            or not info["create_table_sql"].strip()
            or not isinstance(info.get("staging_table"), str)
        ):
            raise ValueError(f"Invalid Redshift schema contract: {name}")
        manifest["tables"][name] = {
            "row_count": info["row_count"],
            "columns": info["columns"],
            "staging_table": info["staging_table"],
            "create_table_sql": info["create_table_sql"],
            "copy_manifest": f"s3://{bucket}/{manifest_key}",
            "files": file_evidence,
        }
    release_key = f"{prefix}/release_manifest.json"
    objects.append(_from_json(release_key, manifest))  # LAST, only after all prior writes
    return {
        "release_id": release_id,
        "bucket": bucket,
        "parquet_count": total_files,
        "parquet_bytes": total_bytes,
        "objects": objects,
        "release_manifest_key": release_key,
    }


def _b64_sha(payload: bytes) -> str:
    return base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")


def _head_matches(s3, bucket: str, item: UploadItem) -> None:
    head = s3.head_object(Bucket=bucket, Key=item.key)
    if (
        head["ContentLength"] != item.size_bytes
        or head.get("Metadata", {}).get("sha256") != item.sha256
    ):
        raise RuntimeError(f"S3 object metadata mismatch: {item.key}")


def _verify_existing(s3, bucket: str, item: UploadItem) -> None:
    """On retry, do NOT trust metadata alone. Stream and hash remote bytes."""
    response = s3.get_object(Bucket=bucket, Key=item.key)
    stream = response["Body"]
    hashed = hashlib.sha256()
    size = 0
    try:
        while chunk := stream.read(BUFFER_SIZE):
            hashed.update(chunk)
            size += len(chunk)
    finally:
        stream.close()
    if size != item.size_bytes or hashed.hexdigest() != item.sha256:
        raise RuntimeError(f"S3 existing object integrity mismatch: {item.key}")
    _head_matches(s3, bucket, item)


def _put_small(s3, bucket: str, item: UploadItem) -> bool:
    """Create-only single PUT; S3 validates full-object SHA-256 in transit."""
    checksum = base64.b64encode(bytes.fromhex(item.sha256)).decode("ascii")
    try:
        with item.open() as stream:
            s3.put_object(
                Bucket=bucket,
                Key=item.key,
                Body=stream,
                ContentLength=item.size_bytes,
                ContentType=item.content_type,
                Metadata={"sha256": item.sha256},
                ChecksumSHA256=checksum,
                IfNoneMatch="*",
            )
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in {"PreconditionFailed", "412"}:
            raise
        return False
    _head_matches(s3, bucket, item)
    return True


def _put_multipart(s3, bucket: str, item: UploadItem) -> bool:
    """Multipart with per-part SHA-256 and conditional *completion*.

    Retains at most one part (64 MiB) in memory. A failed upload is aborted.
    """
    if math.ceil(item.size_bytes / PART_SIZE) > MAX_MULTIPART_PARTS:
        raise ValueError("Too many multipart parts: repartition source Parquet")
    response = s3.create_multipart_upload(
        Bucket=bucket,
        Key=item.key,
        ContentType=item.content_type,
        Metadata={"sha256": item.sha256},
        ChecksumAlgorithm="SHA256",
    )
    upload_id = response["UploadId"]
    completed = False
    try:
        parts = []
        hashed = hashlib.sha256()
        size = 0
        with item.open() as stream:
            for number in range(1, MAX_MULTIPART_PARTS + 1):
                payload = stream.read(PART_SIZE)
                if not payload:
                    break
                hashed.update(payload)
                size += len(payload)
                part_checksum = _b64_sha(payload)
                uploaded = s3.upload_part(
                    Bucket=bucket,
                    Key=item.key,
                    UploadId=upload_id,
                    PartNumber=number,
                    Body=payload,
                    ContentLength=len(payload),
                    ChecksumSHA256=part_checksum,
                )
                parts.append(
                    {
                        "PartNumber": number,
                        "ETag": uploaded["ETag"],
                        "ChecksumSHA256": part_checksum,
                    }
                )
        if size != item.size_bytes or hashed.hexdigest() != item.sha256:
            raise RuntimeError(f"Local file changed during upload: {item.key}")
        try:
            s3.complete_multipart_upload(
                Bucket=bucket,
                Key=item.key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"PreconditionFailed", "412"}:
                return False
            # 409 means a racing request; fail closed, rerun the release.
            raise
        completed = True
        _head_matches(s3, bucket, item)
        return True
    finally:
        if not completed:
            s3.abort_multipart_upload(Bucket=bucket, Key=item.key, UploadId=upload_id)


def put_verified(s3, bucket: str, item: UploadItem) -> str:
    """Create or verify identical existing data; never overwrite conflicting data."""
    created = (
        _put_small(s3, bucket, item)
        if item.size_bytes <= MULTIPART_THRESHOLD
        else _put_multipart(s3, bucket, item)
    )
    if created:
        return "CREATED"
    _verify_existing(s3, bucket, item)
    return "VERIFIED_EXISTING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", default="/opt/airflow/data/delta/serving_exports"
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--bucket", default=os.getenv("NYC_V2_RAW_BUCKET"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    result = build_inventory(Path(args.output_root), args.release_id, args.bucket)
    print("Release:", result["release_id"])
    print("Data files:", result["parquet_count"])
    print("Data MiB:", round(result["parquet_bytes"] / 1024**2, 2))
    print("Objects including manifests:", len(result["objects"]))
    if not args.upload:
        print("R2 DRY RUN: PASS (no AWS calls)")
        return
    region = os.environ.get("AWS_REGION", "eu-central-1")
    if region != "eu-central-1":
        raise ValueError("R2 expected AWS_REGION=eu-central-1")
    s3 = boto3.client("s3", region_name=region)
    for item in result["objects"]:
        status = put_verified(s3, result["bucket"], item)
        print(status, item.key)
    print("R2 S3 STAGING: PASS")


if __name__ == "__main__":
    main()
