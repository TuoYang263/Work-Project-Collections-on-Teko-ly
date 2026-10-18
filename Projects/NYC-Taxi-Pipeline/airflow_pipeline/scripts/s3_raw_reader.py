"""Read and validate S3 raw manifests for Bronze."""

import os
import json
import base64
import hashlib
import tempfile
from pathlib import Path

from scripts.s3_raw_ingestion import (
    build_raw_object_key,
    build_raw_manifest_key,
)
from scripts.load_nyc_yellow_taxi_data import _valid_parquet


def read_raw_manifest(s3_client, reference: dict) -> dict:
    """Verify that the manifest identifies the requested raw version."""

    data_month = reference["data_month"]

    try:
        year_text, month_text = data_month.split("-")
        year = int(year_text)
        month = int(month_text)

        if year < 2009 or data_month != f"{year:04d}-{month:02d}":
            raise ValueError("Invalid data month")

        expected_raw_key = build_raw_object_key(year, month)
        expected_manifest_key = build_raw_manifest_key(year, month)

    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid data month: {data_month!r}") from exc

    if (
        reference["raw_key"] != expected_raw_key
        or reference["manifest_key"] != expected_manifest_key
    ):
        raise ValueError("S3 raw reference key mismatch")

    response = s3_client.get_object(
        Bucket=reference["bucket"],
        Key=expected_manifest_key,
    )

    body = response["Body"]
    try:
        payload = body.read(65537)
    finally:
        body.close()

    if len(payload) > 65536:
        raise ValueError("S3 manifest exceeds size limit")

    try:
        manifest = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid S3 manifest JSON") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Invalid S3 manifest structure")

    metadata = manifest.get("s3")
    verification = manifest.get("verification")

    if not isinstance(metadata, dict) or not isinstance(verification, dict):
        raise ValueError("Invalid S3 manifest structure")

    if (
        manifest.get("schema_version") != 1
        or manifest.get("data_month") != data_month
        or metadata.get("bucket") != reference["bucket"]
        or metadata.get("key") != expected_raw_key
        or metadata.get("version_id") != reference["raw_version_id"]
        or not metadata.get("version_id")
        or not metadata.get("checksum_sha256")
        or type(metadata.get("size_bytes")) is not int
        or metadata["size_bytes"] <= 0
        or metadata.get("encryption") != "AES256"
        or verification.get("result") != "PASS"
    ):
        raise ValueError("S3 manifest does not match requested raw version")

    return manifest


def download_versioned_raw(
    s3_client,
    reference: dict,
    output_dir: str | Path,
) -> str:
    """Download and verify an exact S3 raw object version."""

    manifest = read_raw_manifest(s3_client, reference)
    metadata = manifest["s3"]
    version_id = metadata["version_id"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version_tag = hashlib.sha256(version_id.encode("utf-8")).hexdigest()[:12]

    destination = output_dir / (
        f"yellow_tripdata_{reference['data_month']}_" f"{version_tag}.parquet"
    )

    temp_path = None

    try:
        response = s3_client.get_object(
            Bucket=reference["bucket"],
            Key=reference["raw_key"],
            VersionId=version_id,
        )

        body = response["Body"]

        try:
            if response.get("VersionId") != version_id:
                raise ValueError("S3 returned unexpected raw version")

            checksum = hashlib.sha256()
            downloaded_size = 0

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=output_dir,
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)

                while True:
                    chunk = body.read(1024 * 1024)

                    if not chunk:
                        break

                    temporary.write(chunk)
                    checksum.update(chunk)
                    downloaded_size += len(chunk)

                temporary.flush()
                os.fsync(temporary.fileno())

        finally:
            body.close()

        if downloaded_size != metadata["size_bytes"]:
            raise ValueError("S3 raw size mismatch")

        actual_checksum = base64.b64encode(checksum.digest()).decode("ascii")

        if actual_checksum != metadata["checksum_sha256"]:
            raise ValueError("S3 raw SHA-256 mismatch")

        if not _valid_parquet(temp_path):
            raise ValueError("S3 raw is not a valid NYC TLC Parquet")

        os.replace(temp_path, destination)

        return str(destination)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def resolve_versioned_raw_paths(
    s3_client,
    raw_file_months: list[str],
    raw_references: list[dict],
    output_dir: str | Path,
) -> list[str]:
    """Resolve exact S3 raw versions for Bronze input."""

    if not raw_file_months:
        raise ValueError("Bronze raw file months must not be empty")

    if (
        not isinstance(raw_references, list)
        or len(raw_references) != len(raw_file_months)
        or any(not isinstance(ref, dict) for ref in raw_references)
        or [ref.get("data_month") for ref in raw_references] != raw_file_months
    ):
        raise ValueError("Raw references do not match execution read scope")

    local_paths = []

    for reference in raw_references:
        local_path = download_versioned_raw(
            s3_client=s3_client,
            reference=reference,
            output_dir=output_dir,
        )

        local_paths.append(local_path)

    return local_paths
