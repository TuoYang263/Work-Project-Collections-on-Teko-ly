from scripts import delta_medallion_pipeline as pipeline


def _contract():
    return {
        "mode": "CONFIGURED_FULL_REFRESH",
        "year": 2023,
        "months": [1, 2, 3],
        "window_start": "2023-01-01",
        "window_end_exclusive": "2023-04-01",
        "logical_window": {
            "start": "2023-01-01",
            "end_exclusive": "2023-04-01",
            "anchor": "PICKUP_DATETIME",
        },
        "read_scope": {
            "raw_file_months": [
                "2022-12",
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "boundary_policy":
                "INCLUDE_PREVIOUS_MONTH",
        },
        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics":
            "RECOMPUTE_CONFIGURED_WINDOW",
    }


def test_gold_task_forwards_execution_contract(
    monkeypatch,
):
    recorded = {}

    def fake_write_gold(
        execution_contract,
        **kwargs,
    ):
        recorded["contract"] = execution_contract
        return {"ok": True}

    monkeypatch.setattr(
        pipeline,
        "write_gold",
        fake_write_gold,
    )

    result = pipeline.gold_task(
        execution_contract=_contract(),
    )

    assert recorded["contract"] == _contract()
    assert result == {"ok": True}


def test_bronze_task_forwards_read_scope(
    monkeypatch,
):
    recorded = {}

    def fake_write_bronze(
        raw_file_months,
    ):
        recorded["raw_file_months"] = (
            raw_file_months
        )
        return "bronze-ok"

    monkeypatch.setattr(
        pipeline,
        "write_bronze",
        fake_write_bronze,
    )

    result = pipeline.bronze_task(
        execution_contract=_contract(),
    )

    assert recorded["raw_file_months"] == [
        "2022-12",
        "2023-01",
        "2023-02",
        "2023-03",
    ]

    assert result == "bronze-ok"