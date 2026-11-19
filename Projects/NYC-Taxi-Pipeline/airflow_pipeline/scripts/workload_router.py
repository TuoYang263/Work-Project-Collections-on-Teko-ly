"""NYC V2 rule-based workload routing V1.

Pure decision logic: no Spark, AWS, or job execution.
"""

from collections.abc import Mapping

POLICY_VERSION = "cw-v1.0"

GOLD_PRODUCTS = frozenset(
    {
        "trip_summary_hourly",
        "zone_summary_pickup_daily",
        "zone_summary_dropoff_daily",
        "flow_imbalance_daily",
        "airport_economics_daily",
        "od_corridor_daily",
    }
)


def _certified_manifest_matches(manifest, contract):
    """Check manifest structure and requested window."""

    if not isinstance(manifest, Mapping):
        return False

    if type(manifest.get("schema_version")) is not int:
        return False

    if manifest["schema_version"] != 1:
        return False

    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        return False

    if manifest.get("logical_window") != contract.get("logical_window"):
        return False

    paths = manifest.get("table_paths")
    versions = manifest.get("table_versions")

    if not isinstance(paths, Mapping):
        return False

    if not isinstance(versions, Mapping):
        return False

    if set(paths) != GOLD_PRODUCTS:
        return False

    if set(versions) != GOLD_PRODUCTS:
        return False

    return all(
        isinstance(paths[name], str)
        and bool(paths[name].strip())
        and type(versions[name]) is int
        and versions[name] >= 0
        for name in GOLD_PRODUCTS
    )


def plan_workload(
    *,
    intent: str,
    execution_contract: Mapping,
    active_manifest: Mapping | None = None,
    publication_verified: bool = False,
    freshness_verified: bool = False,
    sources_unchanged_verified: bool = False,
    raw_inputs_ready: bool = False,
    local_resources_ok: bool = False,
    spend_approved: bool = False,
) -> dict:
    """Return a route decision; never launch computation."""

    def decision(route, reason, release_id=None):
        return {
            "policy_version": POLICY_VERSION,
            "selected_route": route,
            "reason_code": reason,
            "release_id": release_id,
            "auto_launch": False,
        }

    if not isinstance(execution_contract, Mapping):
        return decision("BLOCKED_REVIEW", "INVALID_EXECUTION_CONTRACT")

    if intent == "READ_PUBLISHED":
        if not _certified_manifest_matches(active_manifest, execution_contract):
            return decision("BLOCKED_REVIEW", "NO_MATCHING_RELEASE")

        if publication_verified is not True:
            return decision("BLOCKED_REVIEW", "PUBLICATION_UNVERIFIED")

        if freshness_verified is not True:
            return decision("BLOCKED_REVIEW", "FRESHNESS_UNVERIFIED")

        if sources_unchanged_verified is not True:
            return decision("BLOCKED_REVIEW", "SOURCE_STATE_UNVERIFIED")

        return decision(
            "NO_COMPUTE",
            "CERTIFIED_RELEASE_REUSABLE",
            active_manifest["release_id"],
        )

    if intent == "FULL_REFRESH":
        if (
            execution_contract.get("mode") != "CONFIGURED_FULL_REFRESH"
            or execution_contract.get("write_scope") != "FULL_TABLE_OVERWRITE"
        ):
            return decision("BLOCKED_REVIEW", "UNSUPPORTED_WRITE_SCOPE")

        if raw_inputs_ready is not True:
            return decision("BLOCKED_REVIEW", "RAW_INPUTS_NOT_READY")

        if local_resources_ok is not True:
            return decision("BLOCKED_REVIEW", "LOCAL_CAPACITY_UNVERIFIED")

        if spend_approved is not True:
            return decision("BLOCKED_REVIEW", "SPEND_NOT_APPROVED")

        return decision("LOCAL_SPARK", "APPROVED_LOCAL_FULL_REFRESH")

    return decision("BLOCKED_REVIEW", "UNSUPPORTED_INTENT")
