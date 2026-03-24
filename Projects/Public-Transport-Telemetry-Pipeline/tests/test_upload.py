from pathlib import Path
from azure.storage.blob import BlobServiceClient
import base64
import os

connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
container_name = os.environ["AZURE_BLOB_CONTAINER"]

local_file = Path("/tmp/hsl_route_paths.parquet")
blob_name = "telemetry/hsl/test_hsl_route_paths_block.parquet"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

data = local_file.read_bytes()
print(f"[DEBUG] Read complete: {len(data)} bytes")

chunk_size = 256 * 1024  # 256 KB
block_ids = []

print("[DEBUG] Starting staged block upload...")

for i in range(0, len(data), chunk_size):
    chunk = data[i:i + chunk_size]
    block_id = base64.b64encode(f"block-{i // chunk_size:06d}".encode()).decode()
    block_ids.append(block_id)

    print(f"[DEBUG] Staging block {i // chunk_size + 1}, size={len(chunk)} bytes")
    blob_client.stage_block(block_id=block_id, data=chunk)

print("[DEBUG] Committing block list...")
blob_client.commit_block_list(block_ids)

print("[DEBUG] Block upload done")