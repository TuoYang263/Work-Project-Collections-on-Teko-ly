from pathlib import Path
import os
from azure.storage.blob import BlobServiceClient

def upload_file(blob_service_client: BlobServiceClient,
                container_name: str,
                local_path: Path,
                blob_name: str) -> None:
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Uploaded: {local_path} -> {container_name}/{blob_name}")

def main() -> None:
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ["AZURE_BLOB_CONTAINER"]

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "output"

    files_to_upload = [
        "gold_route_daily.parquet",
        "gold_route_window.parquet",
        "pipeline_metrics.parquet",
    ]

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    for filename in files_to_upload:
        local_path = output_dir / filename
        if not local_path.exists():
            raise FileNotFoundError(f"Expected output file not found: {local_path}")
        
        upload_file(
            blob_service_client=blob_service_client,
            container_name=container_name,
            local_path=local_path,
            blob_name=f"telemetry/{filename}",
        )

if __name__ == "__main__":
    main()