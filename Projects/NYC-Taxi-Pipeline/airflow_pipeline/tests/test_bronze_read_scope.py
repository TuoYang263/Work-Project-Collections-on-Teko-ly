import pytest
from scripts import delta_medallion_pipeline as pipeline
from scripts import s3_raw_ingestion, s3_raw_reader


def test_resolve_raw_file_paths_supports_year_boundary(
    monkeypatch,
):
    calls = []

    def fake_download_to_local(
        year,
        months,
    ):
        calls.append((year, months))

        return [f"/tmp/yellow_tripdata_" f"{year}-{months[0]:02d}.parquet"]

    monkeypatch.setattr(
        pipeline,
        "download_to_local",
        fake_download_to_local,
    )

    paths = pipeline.resolve_raw_file_paths(
        [
            "2022-12",
            "2023-01",
        ]
    )

    assert calls == [
        (2022, [12]),
        (2023, [1]),
    ]

    assert paths == [
        "/tmp/yellow_tripdata_2022-12.parquet",
        "/tmp/yellow_tripdata_2023-01.parquet",
    ]


def test_resolve_raw_file_paths_fails_when_required_month_missing(
    monkeypatch,
):
    calls = []

    def fake_download_to_local(
        year,
        months,
    ):
        calls.append((year, months))

        if year == 2022 and months == [12]:
            return []

        return [f"/tmp/yellow_tripdata_" f"{year}-{months[0]:02d}.parquet"]

    monkeypatch.setattr(
        pipeline,
        "download_to_local",
        fake_download_to_local,
    )

    with pytest.raises(
        RuntimeError,
        match=("Required raw file month unavailable: " "2022-12"),
    ):
        pipeline.resolve_raw_file_paths(
            [
                "2022-12",
                "2023-01",
            ]
        )

    assert calls == [
        (2022, [12]),
    ]


def test_bronze_task_forwards_s3_references(monkeypatch):
    calls = {}

    references = [
        {"data_month": "2022-12", "raw_version_id": "v1"},
        {"data_month": "2023-01", "raw_version_id": "v2"},
    ]

    def fake_write_bronze(raw_file_months, raw_references):
        calls["months"] = raw_file_months
        calls["references"] = raw_references
        return "/tmp/bronze"

    monkeypatch.setattr(
        pipeline,
        "write_bronze",
        fake_write_bronze,
    )

    contract = {
        "write_scope": "FULL_TABLE_OVERWRITE",
        "read_scope": {
            "raw_file_months": ["2022-12", "2023-01"],
        },
    }

    result = pipeline.bronze_task(
        execution_contract=contract,
        raw_references=references,
    )

    assert calls["months"] == ["2022-12", "2023-01"]
    assert calls["references"] == references
    assert result == "/tmp/bronze"


def test_bronze_task_rejects_missing_s3_references():
    contract = {
        "write_scope": "FULL_TABLE_OVERWRITE",
        "read_scope": {
            "raw_file_months": ["2024-01"],
        },
    }

    with pytest.raises(
        ValueError,
        match="requires versioned S3 raw references",
    ):
        pipeline.bronze_task(
            execution_contract=contract,
        )


def test_resolve_bronze_s3_paths(monkeypatch):
    calls = {}
    fake_s3 = object()

    def fake_create_client(region, profile):
        assert region == "eu-central-1"
        return fake_s3

    def fake_resolve(**kwargs):
        calls.update(kwargs)
        return ["/tmp/versioned-raw.parquet"]

    monkeypatch.setenv("AWS_REGION", "eu-central-1")

    monkeypatch.setattr(
        s3_raw_ingestion,
        "create_s3_client",
        fake_create_client,
    )

    monkeypatch.setattr(
        s3_raw_reader,
        "resolve_versioned_raw_paths",
        fake_resolve,
    )

    references = [
        {
            "data_month": "2024-01",
            "raw_version_id": "raw-v1",
        }
    ]

    result = pipeline.resolve_bronze_s3_paths(
        raw_file_months=["2024-01"],
        raw_references=references,
    )

    assert result == ["/tmp/versioned-raw.parquet"]
    assert calls["s3_client"] is fake_s3
    assert calls["raw_file_months"] == ["2024-01"]
    assert calls["raw_references"] == references


def test_write_bronze_requires_s3_references():
    with pytest.raises(
        ValueError,
        match="requires versioned S3 raw references",
    ):
        pipeline.write_bronze(
            raw_file_months=["2024-01"],
        )


def test_bronze_reads_s3_resolved_parquet(tmp_path, monkeypatch):
    """Verify S3-resolved input reaches the Delta Bronze table."""

    from datetime import datetime
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_file = tmp_path / "yellow_tripdata_2024-01.parquet"

    table = pa.table(
        {
            "tpep_pickup_datetime": [datetime(2024, 1, 15, 10)],
            "tpep_dropoff_datetime": [datetime(2024, 1, 15, 10, 20)],
            "PULocationID": [100],
            "DOLocationID": [200],
        }
    )

    pq.write_table(table, raw_file)

    references = [
        {
            "data_month": "2024-01",
            "raw_version_id": "test-version-v1",
        }
    ]

    def fake_s3_resolver(raw_file_months, raw_references):
        assert raw_file_months == ["2024-01"]
        assert raw_references == references
        return [str(raw_file)]

    def reject_legacy_downloader(*args, **kwargs):
        raise AssertionError("Bronze used legacy downloader")

    monkeypatch.setattr(
        pipeline,
        "resolve_bronze_s3_paths",
        fake_s3_resolver,
    )

    monkeypatch.setattr(
        pipeline,
        "resolve_raw_file_paths",
        reject_legacy_downloader,
    )

    monkeypatch.setattr(
        pipeline,
        "BRONZE_TRIPS",
        str(tmp_path / "bronze_delta"),
    )

    output_path = pipeline.write_bronze(
        raw_file_months=["2024-01"],
        raw_references=references,
    )

    spark = pipeline.get_spark("Bronze S3 Integration Test")
    bronze = spark.read.format("delta").load(output_path)

    assert bronze.count() == 1
    assert "pickup_day" in bronze.columns
    assert bronze.select("PULocationID").first()[0] == 100
