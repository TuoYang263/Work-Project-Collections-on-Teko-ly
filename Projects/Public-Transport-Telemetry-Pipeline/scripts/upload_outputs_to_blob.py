from pathlib import Path
import os
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

    with open(local_file, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

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

        # rglob("*"): recursively list all files under the directory (including nested ones)
        # filter with is_file() to exclude subdirectories
        files = sorted([p for p in local_path.rglob("*") if p.is_file()])

        if not files:
            raise FileNotFoundError(f"No files found inside directory: {local_path}")
        
        for file_path in files:

            # relative_to(local_path): get path relative to the parquet directory root
            # e.g. /data/.../gold.parquet/part-0000.parquet -> part-0000.parquet
            relative_path = file_path.relative_to(local_path)

            # as_posix(): convert path to POSIX format (use '/' as separator)
            # ensures compatibility with cloud storage paths (Blob/S3)
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

    outputs_to_upload = [
        "gold_route_daily.parquet",
        "gold_route_window.parquet",
        "pipeline_metrics.parquet",
    ]

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    for name in outputs_to_upload:
        local_path = output_dir / name
        if not local_path.exists():
            raise FileNotFoundError(f"Expected output path not found: {local_path}")
        
        blob_prefix = f"telemetry/{name}"

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