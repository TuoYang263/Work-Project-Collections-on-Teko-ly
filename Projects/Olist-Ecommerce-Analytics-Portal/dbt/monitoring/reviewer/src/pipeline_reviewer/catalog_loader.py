from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class CatalogError(ValueError):
    pass


def load_rule_catalog(path: str | Path) -> dict[str, Any]:
    catalog_path = Path(path)

    if not catalog_path.is_file():
        raise CatalogError(f"Rule catalog not found: {catalog_path}")

    try:
        catalog = yaml.safe_load(
            catalog_path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as exc:
        raise CatalogError(
            f"Invalid YAML in rule catalog: {catalog_path}"
        ) from exc

    if not isinstance(catalog, dict):
        raise CatalogError("Rule catalog must be a YAML mapping")

    rules = catalog.get("rules")
    if not isinstance(rules, list):
        raise CatalogError("Rule catalog field 'rules' must be a list")

    rule_ids = [
        rule.get("rule_id")
        for rule in rules
        if isinstance(rule, dict)
    ]

    if len(rule_ids) != len(set(rule_ids)):
        raise CatalogError("Rule catalog contains duplicate rule IDs")

    return catalog