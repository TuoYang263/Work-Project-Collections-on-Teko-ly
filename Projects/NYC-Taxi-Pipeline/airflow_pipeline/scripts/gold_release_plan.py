"""Plan isolated local Gold releases without side effects."""

from hashlib import sha256
from pathlib import Path


def build_gold_release_spec(
    *,
    dag_id: str,
    run_id: str,
    base_dir: str,
) -> dict[str, str]:
    for name, value in {
        "dag_id": dag_id,
        "run_id": run_id,
        "base_dir": base_dir,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    identity = f"{dag_id}\0{run_id}"
    digest = sha256(identity.encode("utf-8")).hexdigest()[:24]
    release_id = f"airflow-{digest}"

    gold_root = Path(base_dir).expanduser().resolve() / "data" / "delta" / "gold"

    return {
        "release_id": release_id,
        "candidate_root": str(gold_root / "releases" / release_id),
        "active_manifest": str(gold_root / "active.json"),
    }
