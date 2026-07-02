FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DBT_PROFILES_DIR=/app/dbt
ENV DBT_TARGET=prod

WORKDIR /app/dbt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        bash \
        ca-certificates \
        gettext-base \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir dbt-bigquery

COPY dbt/ /app/dbt/

RUN chmod +x /app/dbt/run_dbt_job.sh

ENTRYPOINT ["/app/dbt/run_dbt_job.sh"]