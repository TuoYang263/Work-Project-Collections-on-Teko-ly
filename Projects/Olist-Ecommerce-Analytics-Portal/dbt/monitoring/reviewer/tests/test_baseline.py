from __future__ import annotations

import sys
import unittest
from pathlib import Path

REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer import median_baseline  # noqa: E402


class MedianBaselineTests(unittest.TestCase):
    def test_returns_median_for_odd_number_of_values(self) -> None:
        self.assertEqual(
            median_baseline([10, 20, 30]),
            20.0,
        )

    def test_returns_median_for_even_number_of_values(self) -> None:
        self.assertEqual(
            median_baseline([10, 20, 30, 40]),
            25.0,
        )

    def test_ignores_none_values(self) -> None:
        self.assertEqual(
            median_baseline([10, None, 30]),
            20.0,
        )

    def test_returns_none_when_no_numeric_values_exist(self) -> None:
        self.assertIsNone(
            median_baseline([None, None]),
        )

    def test_zero_is_a_valid_baseline_value(self) -> None:
        self.assertEqual(
            median_baseline([0, 0, 0]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
