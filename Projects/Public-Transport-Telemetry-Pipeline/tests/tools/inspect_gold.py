from src.pipeline.setup import build_spark, use_database
from src.pipeline.config import (
    DATABASE_NAME,
    GOLD_ROUTE_WINDOW_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_PIPELINE_METRICS_TABLE,
)

spark = build_spark()
use_database(spark)

print("\nTables:")
spark.sql(f"SHOW TABLES IN {DATABASE_NAME}").show(truncate=False)

print("\nGold route window:")
spark.table(GOLD_ROUTE_WINDOW_TABLE).show(10, truncate=False)
print("count:", spark.table(GOLD_ROUTE_WINDOW_TABLE).count())

print("\nGold route daily:")
spark.table(GOLD_ROUTE_DAILY_TABLE).show(10, truncate=False)
print("count:", spark.table(GOLD_ROUTE_DAILY_TABLE).count())

print("\nGold pipeline metrics:")
spark.table(GOLD_PIPELINE_METRICS_TABLE).show(10, truncate=False)
print("count:", spark.table(GOLD_PIPELINE_METRICS_TABLE).count())