from pathlib import Path
from src.pipeline.hsl import build_hsl_map_outputs

outputs = build_hsl_map_outputs(
    gtfs_dir=Path("data/external/gtfs_hsl"),
    mode="bus",
    route="all",
    lookback_minutes=30,
)

print(outputs["df_map"].shape)
print(outputs["route_options"].shape)
print(outputs["map_points"].shape)
print(outputs["paths"].shape)