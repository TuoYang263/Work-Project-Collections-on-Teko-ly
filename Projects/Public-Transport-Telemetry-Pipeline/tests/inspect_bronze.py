from src.pipeline.setup import build_spark, use_database
from src.pipeline.config import DATABASE_NAME, BRONZE_EVENTS_TABLE

spark = build_spark()
use_database(spark)

print("\nCurrent database:")
spark.sql("SELECT current_database()").show(truncate=False)

print(f"\nTables in {DATABASE_NAME}:")
spark.sql(f"SHOW TABLES IN {DATABASE_NAME}").show(truncate=False)

print("\nBronze schema:")
spark.table(BRONZE_EVENTS_TABLE).printSchema()

print("\nSample rows:")
spark.table(BRONZE_EVENTS_TABLE).show(5, truncate=False)

print("\nRow count:")
print(spark.table(BRONZE_EVENTS_TABLE).count())