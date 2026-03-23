from pathlib import Path
import os
import base64
from azure.storage.blob import BlobServiceClient

def delete_existing_blobs(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_prefix: str,
) -> None:
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs(name_starts_with=blob_prefix))

    if not blobs:
        print(f"No existing blobs found under prefix: {blob_prefix}")
        return

    for blob in blobs:
        print(f"[DEBUG] Deleting existing blob: {blob.name}")
        container_client.delete_blob(blob.name)

    print(f"Deleted {len(blobs)} existing blobs under prefix: {blob_prefix}")

def upload_single_file(
    blob_service_client: BlobServiceClient,
    container_name: str,
    local_file: Path,
    blob_name: str,
) -> None:
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    data = local_file.read_bytes()
    file_size = len(data)
    print(f"[DEBUG] Uploading file size: {file_size / (1024 * 1024):.2f} MB")

    # Use staged block upload for better reliability in current environment.
    chunk_size = 256 * 1024  # 256 KB
    block_ids = []

    for i in range(0, file_size, chunk_size):
        chunk = data[i:i + chunk_size]
        block_id = base64.b64encode(f"block-{i // chunk_size:06d}".encode()).decode()
        block_ids.append(block_id)

        print(
            f"[DEBUG] Staging block {i // chunk_size + 1}, "
            f"size={len(chunk)} bytes -> {blob_name}"
        )
        blob_client.stage_block(block_id=block_id, data=chunk)

    print(f"[DEBUG] Committing block list -> {blob_name}")
    blob_client.commit_block_list(block_ids)

    print(f"Uploaded: {local_file} -> {container_name}/{blob_name}")

def upload_path(
    blob_service_client: BlobServiceClient,
    container_name: str,
    local_path: Path,
    blob_prefix: str,
) -> None:
    if local_path.is_file():
        upload_single_file(
            blob_service_client=blob_service_client,
            container_name=container_name,
            local_file=local_path,
            blob_name=blob_prefix,
        )
        return

    if local_path.is_dir():
        files = sorted([p for p in local_path.rglob("*") if p.is_file()])

        if not files:
            raise FileNotFoundError(f"No files found inside directory: {local_path}")

        for file_path in files:
            relative_path = file_path.relative_to(local_path)
            blob_name = f"{blob_prefix}/{relative_path.as_posix()}"

            print(f"[DEBUG] Uploading {file_path} -> {blob_name}")

            upload_single_file(
                blob_service_client=blob_service_client,
                container_name=container_name,
                local_file=file_path,
                blob_name=blob_name,
            )
        return

    raise FileNotFoundError(f"Expected file or directory not found: {local_path}")


def main() -> None:
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ["AZURE_BLOB_CONTAINER"]

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "output"
    hsl_dir = project_root / "data" / "gold" / "hsl"
    weather_dir = project_root / "data" / "gold" / "weather"

    outputs_to_upload = [
        ("telemetry/gold_route_daily.parquet", output_dir / "gold_route_daily.parquet"),
        ("telemetry/gold_route_window.parquet", output_dir / "gold_route_window.parquet"),
        ("telemetry/pipeline_metrics.parquet", output_dir / "pipeline_metrics.parquet"),
        ("telemetry/hsl/hsl_df_map.parquet", hsl_dir / "hsl_df_map.parquet"),
        ("telemetry/hsl/hsl_map_points.parquet", hsl_dir / "hsl_map_points.parquet"),
        ("telemetry/hsl/hsl_route_paths.parquet", hsl_dir / "hsl_route_paths.parquet"),
        ("telemetry/hsl/hsl_route_options.parquet", hsl_dir / "hsl_route_options.parquet"),
        ("telemetry/weather/weather_stations_latest.parquet", weather_dir / "weather_stations_latest.parquet"),
    ]

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    for blob_prefix, local_path in outputs_to_upload:
        if not local_path.exists():
            raise FileNotFoundError(f"Expected output path not found: {local_path}")

        delete_existing_blobs(
            blob_service_client=blob_service_client,
            container_name=container_name,
            blob_prefix=blob_prefix,
        )

        upload_path(
            blob_service_client=blob_service_client,
            container_name=container_name,
            local_path=local_path,
            blob_prefix=blob_prefix,
        )


if __name__ == "__main__":
    main()