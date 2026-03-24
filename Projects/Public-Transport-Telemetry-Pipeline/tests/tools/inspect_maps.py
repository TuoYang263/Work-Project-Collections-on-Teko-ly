from __future__ import annotations

from pathlib import Path
import pprint

from streamlit_app.utils.maps import (
    load_map_parquets,
    get_route_options,
    prepare_points_for_pydeck,
    prepare_paths_for_pydeck,
    build_map_bundle,
)


# Adjust this if your exported map parquet files are stored elsewhere
MAP_DATA_DIR = Path("data/gold/hsl")


def print_df_info(name, df, n=5):
    print(f"\n{'=' * 80}")
    print(f"[{name}]")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(n).to_string())
    else:
        print("(empty dataframe)")


def main():
    print("\nStarting maps.py inspection...")
    print(f"MAP_DATA_DIR = {MAP_DATA_DIR.resolve()}")

    # ------------------------------------------------------------------
    # 1. Load raw parquet assets
    # ------------------------------------------------------------------
    loaded = load_map_parquets(MAP_DATA_DIR)

    print("\nLoaded keys:")
    pprint.pprint(list(loaded.keys()))

    df_map = loaded.get("df_map")
    map_points = loaded.get("map_points")
    paths = loaded.get("paths")
    route_options = loaded.get("route_options")

    print_df_info("df_map", df_map)
    print_df_info("map_points", map_points)
    print_df_info("paths", paths)
    print_df_info("route_options", route_options)

    # ------------------------------------------------------------------
    # 2. Test route options
    # ------------------------------------------------------------------
    fallback_df = df_map if df_map is not None and not df_map.empty else map_points
    routes = get_route_options(route_options, fallback_df=fallback_df)

    print(f"\n{'=' * 80}")
    print("[route options]")
    print(f"count: {len(routes)}")
    print(f"sample: {routes[:10]}")

    # ------------------------------------------------------------------
    # 3. Test point preparation
    # ------------------------------------------------------------------
    points_layer_df = prepare_points_for_pydeck(map_points)

    print_df_info("points_layer_df", points_layer_df)

    if not points_layer_df.empty:
        required_point_cols = {"lat", "lon", "route_label"}
        missing = required_point_cols - set(points_layer_df.columns)
        if missing:
            print(f"WARNING: points_layer_df is missing columns: {missing}")
        else:
            print("OK: points_layer_df contains lat/lon/route_label")

    # ------------------------------------------------------------------
    # 4. Test path preparation
    # ------------------------------------------------------------------
    path_layer_df = prepare_paths_for_pydeck(paths)

    print_df_info("path_layer_df", path_layer_df)

    if not path_layer_df.empty:
        if "path" not in path_layer_df.columns:
            print("WARNING: path_layer_df does not contain 'path' column")
        else:
            print("OK: path_layer_df contains 'path' column")
            print("\nFirst path sample:")
            print(path_layer_df["path"].iloc[0])

    # ------------------------------------------------------------------
    # 5. Test high-level bundle
    # ------------------------------------------------------------------
    bundle = build_map_bundle(MAP_DATA_DIR, selected_route=None)

    print(f"\n{'=' * 80}")
    print("[build_map_bundle output keys]")
    pprint.pprint(list(bundle.keys()))

    print("\n[center]")
    pprint.pprint(bundle["center"])

    print("\n[points summary]")
    print(bundle["points"].shape)
    print(bundle["points"].head().to_string() if not bundle["points"].empty else "(empty)")

    print("\n[paths summary]")
    print(bundle["paths"].shape)
    print(bundle["paths"].head().to_string() if not bundle["paths"].empty else "(empty)")

    print("\n[routes summary]")
    print(f"count: {len(bundle['routes'])}")
    print(f"sample: {bundle['routes'][:10]}")

    # ------------------------------------------------------------------
    # 6. Optional: test one selected route
    # ------------------------------------------------------------------
    if bundle["routes"]:
        selected_route = bundle["routes"][0]
        print(f"\n{'=' * 80}")
        print(f"[route filter test] selected_route = {selected_route}")

        route_bundle = build_map_bundle(MAP_DATA_DIR, selected_route=selected_route)

        print("filtered points shape:", route_bundle["points"].shape)
        print("filtered paths shape:", route_bundle["paths"].shape)
        print("filtered center:", route_bundle["center"])

    print(f"\n{'=' * 80}")
    print("maps.py inspection finished.")


if __name__ == "__main__":
    main()