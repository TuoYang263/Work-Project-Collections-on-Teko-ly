from collections import Counter
from pathlib import Path
import sys
import yaml

DEFAULT_PATH = Path("dbt/monitoring/reviewer/config/rule_catalog.yml")


def validate(path: Path) -> None:
    raw_text = path.read_text(encoding="utf-8")

    assert raw_text.endswith("\n"), "File must end with a newline"
    assert "\t" not in raw_text, "Tabs are not allowed"

    trailing_whitespace_lines = [
        line_number
        for line_number, line in enumerate(raw_text.splitlines(), start=1)
        if line.rstrip() != line
    ]
    assert not trailing_whitespace_lines, (
        "Trailing whitespace found on lines: "
        f"{trailing_whitespace_lines}"
    )

    catalog = yaml.safe_load(raw_text)
    assert isinstance(catalog, dict), "Catalog must be a mapping"
    assert catalog.get("catalog_version") == "1.0.0"

    policy = catalog.get("evaluation_policy")
    assert isinstance(policy, dict), "evaluation_policy must be a mapping"
    assert policy["missing_required_evidence_result"] == "NOT_EVALUATED"
    assert policy["deterministic_rules_are_source_of_truth"] is True
    assert policy["findings_are_immutable_to_llm"] is True
    assert policy["severity_is_immutable_to_llm"] is True
    assert policy["anomaly_is_not_automatically_error"] is True

    rules = catalog.get("rules")
    assert isinstance(rules, list), "rules must be a list"

    expected_rule_ids = {
        "M9-R001", "M9-R002", "M9-R003",
        "M9-R004", "M9-R005", "M9-R006",
    }
    rule_ids = [rule["rule_id"] for rule in rules]
    duplicates = [
        rule_id
        for rule_id, count in Counter(rule_ids).items()
        if count > 1
    ]
    assert not duplicates, f"Duplicate rule IDs: {duplicates}"
    assert set(rule_ids) == expected_rule_ids, (
        f"Expected {sorted(expected_rule_ids)}, found {sorted(rule_ids)}"
    )

    required_rule_fields = {
        "rule_id", "version", "name", "description", "dimensions",
        "default_severity", "applicability", "required_evidence",
        "trigger_logic", "can_prove", "cannot_prove", "risk",
        "recommendation", "implementation_status",
    }
    rules_by_id = {rule["rule_id"]: rule for rule in rules}

    expected_implementation_status = {
        "M9-R001": "IMPLEMENTED",
        "M9-R002": "IMPLEMENTED",
        "M9-R003": "IMPLEMENTED",
        "M9-R004": "DEFINED_NOT_IMPLEMENTED",
        "M9-R005": "DEFINED_NOT_IMPLEMENTED",
        "M9-R006": "DEFINED_NOT_IMPLEMENTED",
    }

    for rule_id, rule in rules_by_id.items():
        missing_fields = required_rule_fields - set(rule)
        assert not missing_fields, (
            f"{rule_id} missing fields: {sorted(missing_fields)}"
        )
        assert rule["version"] == "1.0.0"
        assert rule["implementation_status"] == (
            expected_implementation_status[rule_id]
        )
        assert rule["default_severity"] in {
            "LOW", "MEDIUM", "HIGH", "CRITICAL"
        }
        assert isinstance(rule["dimensions"], list) and rule["dimensions"]
        assert isinstance(rule["required_evidence"], list)
        assert isinstance(rule["can_prove"], list) and rule["can_prove"]
        assert isinstance(rule["cannot_prove"], list) and rule["cannot_prove"]

    r003 = rules_by_id["M9-R003"]
    r003_evidence = r003["required_evidence"][0]
    assert r003["name"] == "Test Result Non-Passing"
    assert r003_evidence["source"] == "test_run_results"
    assert set(r003_evidence["fields"]) == {
        "monitoring_run_id", "unique_id", "test_name", "status"
    }
    assert set(r003_evidence["context_fields"]) == {
        "model_unique_id", "model_name", "test_type",
        "test_metadata_name", "column_name", "severity", "failures",
        "message", "adapter_response_json",
    }
    assert r003["trigger_logic"] == {
        "field": "test_run_results.status",
        "normalization": "trim_and_lowercase",
        "operator": "not_equals",
        "comparison_value": "pass",
    }
    assert r003["severity_logic"] == {
        "field": "test_run_results.status",
        "normalization": "trim_and_lowercase",
        "mapping": {"warn": "MEDIUM"},
        "fallback": "HIGH",
    }

    r004 = rules_by_id["M9-R004"]
    assert r004["name"] == "Model Missing from Current Run"
    assert r004["baseline_policy"]["selection"] == (
        "immediately_previous_comparable_run"
    )
    assert r004["trigger_logic"]["operation"] == "set_difference"
    assert r004["trigger_logic"]["operator"] == (
        "baseline_minus_current_not_empty"
    )
    assert set(r004["trigger_logic"]["filters"]["resource_type_in"]) == {
        "model"
    }

    r005 = rules_by_id["M9-R005"]
    assert r005["name"] == "Row-Count Anomaly"
    assert r005["baseline_policy"]["window_size"] == 5
    assert r005["baseline_policy"]["aggregation"] == "median"
    assert r005["trigger_logic"]["relative_change_threshold"] == 0.30
    assert r005["trigger_logic"]["minimum_absolute_change"] == 100
    assert r005["trigger_logic"]["threshold_combination"] == "all"

    r006 = rules_by_id["M9-R006"]
    assert r006["name"] == "Runtime Regression"
    assert r006["baseline_policy"]["window_size"] == 5
    assert r006["baseline_policy"]["aggregation"] == "median"
    assert r006["trigger_logic"]["relative_increase_threshold"] == 0.50
    assert r006["trigger_logic"]["minimum_absolute_increase_seconds"] == 5.0
    assert r006["trigger_logic"]["threshold_combination"] == "all"
    assert r006["trigger_logic"]["zero_or_negative_baseline_result"] == (
        "NOT_EVALUATED"
    )

    allowed_sources = {
        "pipeline_runs", "model_run_results", "test_run_results",
        "model_metadata_snapshots", "model_column_snapshots",
        "model_lineage_edges",
    }
    for rule in rules:
        for evidence in rule["required_evidence"]:
            assert evidence["source"] in allowed_sources, (
                f"{rule['rule_id']} references unknown source "
                f"{evidence['source']}"
            )

    print(f"Validated: {path}")
    print("YAML parsing passed.")
    print("Whitespace and final-newline checks passed.")
    print("Top-level evaluation policy validation passed.")
    print("Rule ID uniqueness and R001-R006 completeness passed.")
    print("R003 evidence, trigger, and severity validation passed.")
    print("R004 model-inventory comparison validation passed.")
    print("R005 row-count baseline and threshold validation passed.")
    print("R006 runtime baseline and threshold validation passed.")
    print("All referenced evidence sources are valid M8 monitoring tables.")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    validate(target)
