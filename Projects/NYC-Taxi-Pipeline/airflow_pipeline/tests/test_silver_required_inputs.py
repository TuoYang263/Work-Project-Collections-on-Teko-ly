"""Guardrails for Silver inputs consumed by Gold."""

import pytest

from scripts import transform_trip_data_spark as transforms


class FakeFrame:
    def __init__(self, columns, rows=1):
        self.columns = list(columns)
        self.rows = rows

    def limit(self, n):
        return FakeFrame(self.columns, min(self.rows, n))

    def count(self):
        return self.rows


@pytest.fixture
def gold_fields():
    return sorted(transforms.REQUIRED_GOLD_INPUT_COLUMNS)


def test_all_required_columns_accepted(gold_fields):
    transforms.require_gold_input_columns(FakeFrame(gold_fields))


def test_missing_required_column_rejected(gold_fields):
    gold_fields.remove("tip_amount")
    with pytest.raises(ValueError, match="tip_amount"):
        transforms.require_gold_input_columns(FakeFrame(gold_fields))


def test_empty_cleaned_data_rejected(gold_fields):
    with pytest.raises(ValueError, match="no rows"):
        transforms.require_nonempty_silver(FakeFrame(gold_fields, rows=0))


def test_nonempty_cleaned_data_accepted(gold_fields):
    transforms.require_nonempty_silver(FakeFrame(gold_fields, rows=12))


def test_transform_rejects_missing_before_write(monkeypatch, gold_fields):
    # The cleaning operations are replaced here; test the stop condition.
    missing_tip = FakeFrame([x for x in gold_fields if x != "tip_amount"])
    monkeypatch.setattr(transforms, "clean_column_names", lambda df: df)
    monkeypatch.setattr(transforms, "cast_column_types", lambda df: df)
    monkeypatch.setattr(transforms, "drop_useless_columns", lambda df: df)
    monkeypatch.setattr(transforms, "filter_invalid_rows", lambda df: df)

    with pytest.raises(ValueError, match="tip_amount"):
        transforms.transform_trip_data(missing_tip)


def test_transform_rejects_empty_after_filtering(monkeypatch, gold_fields):
    original = FakeFrame(gold_fields)
    monkeypatch.setattr(transforms, "clean_column_names", lambda df: df)
    monkeypatch.setattr(transforms, "cast_column_types", lambda df: df)
    monkeypatch.setattr(transforms, "drop_useless_columns", lambda df: df)
    monkeypatch.setattr(
        transforms,
        "filter_invalid_rows",
        lambda df: FakeFrame(df.columns, rows=0),
    )

    with pytest.raises(ValueError, match="no rows"):
        transforms.transform_trip_data(original)
