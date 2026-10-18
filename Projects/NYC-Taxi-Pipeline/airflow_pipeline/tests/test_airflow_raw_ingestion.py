import pytest

from scripts import s3_raw_ingestion
from scripts import airflow_raw_ingestion
from scripts import load_nyc_yellow_taxi_data as downloader
from scripts.airflow_raw_ingestion import extract_raw_months


def test_extract_raw_months_cross_year():
    contract = {
        "mode": "CONFIGURED_FULL_REFRESH",
        "read_scope": {"raw_file_months": ["2022-12", "2023-01", "2023-02"]},
    }

    assert extract_raw_months(contract) == [
        "2022-12",
        "2023-01",
        "2023-02",
    ]


def test_rejects_backfill():
    contract = {
        "mode": "BACKFILL",
        "read_scope": {"input_layer": "SILVER"},
    }

    with pytest.raises(ValueError, match="CONFIGURED_FULL_REFRESH"):
        extract_raw_months(contract)


def test_rejects_invalid_month():
    contract = {
        "mode": "CONFIGURED_FULL_REFRESH",
        "read_scope": {"raw_file_months": ["2024-13"]},
    }

    with pytest.raises(ValueError, match="Invalid raw file month"):
        extract_raw_months(contract)


def test_stage_raw_months_cross_year(tmp_path, monkeypatch):
    calls = []

    def fake_download(year, months):
        calls.append((year, months))

        filename = f"yellow_tripdata_{year}-{months[0]:02d}.parquet"
        path = tmp_path / filename
        path.write_bytes(b"mock data")

        return [str(path)]

    monkeypatch.setattr(downloader, "download_to_local", fake_download)

    contract = {
        "mode": "CONFIGURED_FULL_REFRESH",
        "read_scope": {"raw_file_months": ["2022-12", "2023-01"]},
    }

    result = airflow_raw_ingestion.stage_raw_months(contract)

    assert calls == [
        (2022, [12]),
        (2023, [1]),
    ]

    assert [item["data_month"] for item in result] == [
        "2022-12",
        "2023-01",
    ]


def test_stage_raw_months_rejects_wrong_file(tmp_path, monkeypatch):
    wrong_file = tmp_path / "yellow_tripdata_2024-02.parquet"
    wrong_file.write_bytes(b"mock data")

    monkeypatch.setattr(
        downloader,
        "download_to_local",
        lambda year, months: [str(wrong_file)],
    )

    contract = {
        "mode": "CONFIGURED_FULL_REFRESH",
        "read_scope": {"raw_file_months": ["2024-01"]},
    }

    with pytest.raises(ValueError, match="month mismatch"):
        airflow_raw_ingestion.stage_raw_months(contract)


def test_ingest_staged_months_calls_s3(monkeypatch):
    calls = []
    fake_s3 = object()

    def fake_ingest_month(**kwargs):
        calls.append(kwargs)

        return {
            "data_month": f"{kwargs['year']}-{kwargs['month']:02d}",
            "raw_status": "REUSED",
            "manifest_status": "REUSED",
        }

    monkeypatch.setattr(
        s3_raw_ingestion,
        "ingest_month",
        fake_ingest_month,
    )

    staged_files = [
        {
            "data_month": "2022-12",
            "local_path": "/tmp/yellow_tripdata_2022-12.parquet",
        },
        {
            "data_month": "2023-01",
            "local_path": "/tmp/yellow_tripdata_2023-01.parquet",
        },
    ]

    results = airflow_raw_ingestion.ingest_staged_months(
        staged_files,
        fake_s3,
        "test-bucket",
    )

    assert len(results) == 2
    assert calls[0]["year"] == 2022
    assert calls[0]["month"] == 12
    assert calls[1]["year"] == 2023
    assert calls[1]["month"] == 1
    assert all(call["s3_client"] is fake_s3 for call in calls)
    assert all(call["bucket"] == "test-bucket" for call in calls)


def test_ingest_staged_months_stops_on_failure(monkeypatch):
    calls = []

    def fake_ingest_month(**kwargs):
        calls.append(kwargs["month"])

        if kwargs["month"] == 1:
            raise RuntimeError("S3 ingestion failed")

        return {"raw_status": "UPLOADED"}

    monkeypatch.setattr(
        s3_raw_ingestion,
        "ingest_month",
        fake_ingest_month,
    )

    staged_files = [
        {"data_month": "2024-01", "local_path": "/tmp/jan.parquet"},
        {"data_month": "2024-02", "local_path": "/tmp/feb.parquet"},
    ]

    with pytest.raises(RuntimeError, match="S3 ingestion failed"):
        airflow_raw_ingestion.ingest_staged_months(
            staged_files,
            object(),
            "test-bucket",
        )

    assert calls == [1]


def test_run_raw_ingestion_returns_xcom_references(monkeypatch):
    monkeypatch.setenv("NYC_V2_RAW_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_REGION", "eu-central-1")

    fake_s3 = object()

    staged = [
        {
            "data_month": "2024-01",
            "local_path": "/tmp/yellow_tripdata_2024-01.parquet",
        }
    ]

    def fake_create_client(**kwargs):
        assert kwargs["region"] == "eu-central-1"
        return fake_s3

    def fake_ingest(staged_files, s3_client, bucket):
        assert staged_files == staged
        assert s3_client is fake_s3
        assert bucket == "test-bucket"

        return [
            {
                "data_month": "2024-01",
                "raw_key": "raw/test.parquet",
                "raw_version_id": "raw-v1",
                "manifest_key": "raw/test.parquet.manifest.json",
                "raw_status": "REUSED",
                "manifest_status": "REUSED",
            }
        ]

    monkeypatch.setattr(
        airflow_raw_ingestion,
        "stage_raw_months",
        lambda contract: staged,
    )

    monkeypatch.setattr(
        s3_raw_ingestion,
        "create_s3_client",
        fake_create_client,
    )

    monkeypatch.setattr(
        airflow_raw_ingestion,
        "ingest_staged_months",
        fake_ingest,
    )

    result = airflow_raw_ingestion.run_raw_ingestion(
        {"mode": "CONFIGURED_FULL_REFRESH"}
    )

    assert result[0]["bucket"] == "test-bucket"
    assert result[0]["raw_version_id"] == "raw-v1"
    assert result[0]["manifest_key"] == ("raw/test.parquet.manifest.json")

    assert "local_path" not in result[0]

    # Airflow's default JSON XCom can serialize this result.
    import json

    json.dumps(result)


def test_run_raw_ingestion_requires_bucket(monkeypatch):
    monkeypatch.delenv("NYC_V2_RAW_BUCKET", raising=False)

    with pytest.raises(
        ValueError,
        match="NYC_V2_RAW_BUCKET is not configured",
    ):
        airflow_raw_ingestion.run_raw_ingestion({"mode": "CONFIGURED_FULL_REFRESH"})
