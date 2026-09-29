import pytest
import inspect

from scripts import delta_medallion_pipeline as pipeline


class FakeWriter:
    def __init__(self):
        self.actions = []

    def mode(self, value):
        self.actions.append(
            ("mode", value)
        )
        return self

    def format(self, value):
        self.actions.append(
            ("format", value)
        )
        return self

    def save(self, path):
        self.actions.append(
            ("save", path)
        )

    def option(self, key, value):
        self.actions.append(
            ("option", key, value)
        )
        return self


class FakeDataFrame:
    def __init__(self):
        self.writer = FakeWriter()

    @property
    def write(self):
        return self.writer


def test_delta_writer_uses_full_table_overwrite():
    df = FakeDataFrame()

    result = pipeline.write_delta_table(
        df,
        target_path="/tmp/gold_test",
        execution_contract={
            "write_scope": "FULL_TABLE_OVERWRITE",
        },
        label="test_table",
    )

    assert result == "/tmp/gold_test"

    assert df.writer.actions == [
        ("mode", "overwrite"),
        ("format", "delta"),
        ("save", "/tmp/gold_test"),
    ]


def test_delta_writer_rejects_unsupported_scope():
    df = FakeDataFrame()

    with pytest.raises(
        ValueError,
        match="Unsupported write_scope",
    ):
        pipeline.write_delta_table(
            df,
            target_path="/tmp/gold_test",
            execution_contract={
                "write_scope": "UNKNOWN_SCOPE",
            },
            label="test_table",
        )

    assert df.writer.actions == []


def test_gold_write_path_does_not_bypass_delta_writer():
    source = inspect.getsource(
        pipeline.write_gold
    )

    assert ".write" not in source


def test_delta_writer_uses_window_replace():
    df = FakeDataFrame()

    result = pipeline.write_delta_table(
        df,
        target_path="/tmp/gold_test",
        execution_contract={
            "write_scope": "WINDOW_REPLACE",
            "logical_window": {
                "start": "2023-02-01",
                "end_exclusive": "2023-03-01",
            },
        },
        label="test_table",
        window_column="service_day",
    )

    assert result == "/tmp/gold_test"

    assert df.writer.actions == [
        ("mode", "overwrite"),
        ("format", "delta"),
        (
            "option",
            "replaceWhere",
            "service_day >= '2023-02-01' "
            "AND service_day < '2023-03-01'",
        ),
        ("save", "/tmp/gold_test"),
    ]


def test_window_replace_requires_window_column():
    df = FakeDataFrame()

    with pytest.raises(
        ValueError,
        match="WINDOW_REPLACE requires window_column",
    ):
        pipeline.write_delta_table(
            df,
            target_path="/tmp/gold_test",
            execution_contract={
                "write_scope": "WINDOW_REPLACE",
                "logical_window": {
                    "start": "2023-02-01",
                    "end_exclusive": "2023-03-01",
                },
            },
            label="test_table",
        )

    assert df.writer.actions == []


def test_gold_window_columns_match_business_time():
    assert pipeline.GOLD_WINDOW_COLUMNS == {
        "trip_summary_hourly": "pickup_hour",
        "zone_summary_pickup_daily":
            "pickup_day",
        "zone_summary_dropoff_daily":
            "dropoff_day",
        "flow_imbalance_daily":
            "service_day",
        "airport_economics_daily":
            "service_day",
        "od_corridor_daily":
            "service_day",
    }