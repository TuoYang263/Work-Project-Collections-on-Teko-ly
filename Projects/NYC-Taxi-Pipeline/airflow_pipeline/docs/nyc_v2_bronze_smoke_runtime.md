# NYC V2 — S3-to-Bronze Smoke Runtime

## Purpose

Validate Airflow CeleryExecutor -> AWS S3 versioned raw
-> PySpark -> isolated Delta Bronze -> Run Evidence V2.

This is a local integration test, not a production deployment.

## Tested baseline

- DAG: `nyc_bronze_s3_smoke`
- Run ID: `bronze_smoke_20261021_01`
- Source: NYC Yellow Taxi, 2024-01
- Bronze rows: 2,964,624
- Bronze columns: 21
- Run Evidence V2: SUCCEEDED
- Original Bronze table: protected by an isolated bind mount

## Compose configuration

Base:
`docker-compose.lifecycle-test.yaml`

Local smoke override:
`docker-compose.bronze-smoke.yaml`

Always use both files when recreating the smoke Worker.

Validate configuration:

    docker compose -f docker-compose.lifecycle-test.yaml \
      -f docker-compose.bronze-smoke.yaml config --quiet

## Local prerequisites

The following private directories must exist on WSL ext4:

- `~/.nyc-v2-worker-auth`
- `~/.nyc-v2-worker-auth/cache`
- `~/.nyc-v2-bronze-smoke/delta`

The AWS configuration uses profile `nyc-v2-dev`.
Host and Worker use separate login caches. Host cache files belong to UID 1000 (0600); Worker cache files belong to UID 50000 (0600). Do not share a writable cache between the two users.

Do not commit AWS credentials, login cache files,
session tokens, or generated Delta data.

## Runtime safety

- AWS authentication uses boto3 LoginProvider.
- No static AWS access keys are injected into Worker.
- The AWS login cache contains sensitive session material.
- The Bronze output is bind-mounted to isolated WSL storage.
- The primary NYC medallion DAG must remain paused.
- Spark uses local[2] with a 1g driver configuration.
- These Spark settings are not a hard container memory limit.

## Re-running the smoke test

The smoke DAG intentionally refuses to overwrite an
existing Delta table in the isolated smoke target.

Do not delete the original Bronze table or blindly
clear the smoke directory to rerun the DAG.

## Remaining validation

- Confirm credential renewal across token expiration.
- Check cache file ACLs after credential refresh.
- Validate recovery after restarting Docker and WSL.
- Extend integration validation to Silver, Gold and Quality.
- Keep AI Incident Investigator read-only initially.

## Credential lifecycle validation

Validated on 2026-10-25:

- Host interactive login uses a dedicated ext4 cache:
  `~/.nyc-v2-host-auth/cache`.
- Worker uses `~/.nyc-v2-worker-auth/cache`.
- Cache contents were synchronized while Worker was stopped.
- Worker cache files are owned by UID 50000, mode 0600.
- Host cache files are owned by UID 1000, mode 0600.
- Native boto3 LoginProvider authentication passed.
- Old credential expiry: 13:33:44 UTC.
- New credential expiry: 13:52:18 UTC.
- Post-expiry AWS STS authentication passed.
- Host authentication also passed after Worker refresh.

The test used a new Python process within the same Worker
container. Long-lived process refresh is not yet verified.

Host and Worker caches originate from the same AWS login session.
Separate filesystem ownership does not guarantee independent
server-side session state.

### Local recovery procedure

1. Reauthenticate through the Host-only AWS profile/cache.
2. Stop the Worker before synchronizing session cache contents.
3. Validate cache filenames and copy only the required session
   file into the Worker cache.
4. Set Worker cache ownership to UID 50000 and mode 0600.
5. Start Worker using both Compose files.
6. Verify LoginProvider, AWS identity, and cache permissions.

Never commit credential caches or print session material.
Production AWS deployments should use IAM roles instead.
