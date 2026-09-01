from collections.abc import Iterable
from statistics import median


def median_baseline(
    values: Iterable[int | float | None],
) -> float | None:
    """Return the median of available numeric baseline values."""

    usable_values = [float(value) for value in values if value is not None]

    if not usable_values:
        return None

    return float(median(usable_values))
