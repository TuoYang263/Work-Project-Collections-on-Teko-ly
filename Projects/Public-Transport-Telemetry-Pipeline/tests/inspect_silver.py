from src.pipeline.setup import build_spark, use_database
from src.pipeline.config import DATABASE_NAME, SILVER_TRANSIT_TABLE, SILVER_WEATHER_TABLE

spark = build_spark()
use_database(spark)

print("\nTables:")
spark.sql(f"SHOW TABLES IN {DATABASE_NAME}").show(truncate=False)

print("\nSilver transit:")
spark.table(SILVER_TRANSIT_TABLE).show(10, truncate=False)
print("count:", spark.table(SILVER_TRANSIT_TABLE).count())

print("\nSilver weather:")
spark.table(SILVER_WEATHER_TABLE).show(10, truncate=False)
print("count:", spark.table(SILVER_WEATHER_TABLE).count())