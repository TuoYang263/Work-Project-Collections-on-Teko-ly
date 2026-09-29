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
                "write_scope": "WINDOW_REPLACE",
            },
            label="test_table",
        )

    assert df.writer.actions == []


def test_gold_write_path_does_not_bypass_delta_writer():
    source = inspect.getsource(
        pipeline.write_gold
    )

    assert ".write" not in source