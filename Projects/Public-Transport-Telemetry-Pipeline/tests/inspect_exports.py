from src.pipeline.setup import build_spark, setup_logging, ensure_directories, use_database
from src.pipeline.config import (
    GOLD_ROUTE_WINDOW_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_PIPELINE_METRICS_TABLE,
    GOLD_ROUTE_WINDOW_EXPORT_PATH,
    GOLD_ROUTE_DAILY_EXPORT_PATH,
    GOLD_PIPELINE_METRICS_EXPORT_PATH,
)

ensure_directories()
logger = setup_logging("inspect_exports")
spark = build_spark("inspect_exports")

try:
    use_database(spark)

    checks = [
        (GOLD_ROUTE_WINDOW_TABLE, GOLD_ROUTE_WINDOW_EXPORT_PATH),
        (GOLD_ROUTE_DAILY_TABLE, GOLD_ROUTE_DAILY_EXPORT_PATH),
        (GOLD_PIPELINE_METRICS_TABLE, GOLD_PIPELINE_METRICS_EXPORT_PATH),
    ]

    for table_name, export_path in checks:
        table_count = spark.table(table_name).count()
        export_count = spark.read.parquet(str(export_path)).count()

        print(f"{table_name}: table_count={table_count}, export_count={export_count}")

finally:
    spark.stop()