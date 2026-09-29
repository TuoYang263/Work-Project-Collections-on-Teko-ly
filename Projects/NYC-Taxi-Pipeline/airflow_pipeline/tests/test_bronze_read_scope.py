from scripts import delta_medallion_pipeline as pipeline


def test_resolve_raw_file_paths_supports_year_boundary(
    monkeypatch,
):
    calls = []

    def fake_download_to_local(
        year,
        months,
    ):
        calls.append(
            (year, months)
        )

        return [
            f"/tmp/yellow_tripdata_"
            f"{year}-{months[0]:02d}.parquet"
        ]

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