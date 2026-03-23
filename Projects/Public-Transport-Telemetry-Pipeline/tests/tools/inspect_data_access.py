from streamlit_app.utils.data_access import (
    load_hsl_df_map,
    load_hsl_map_points,
    load_hsl_route_paths,
    load_hsl_route_options,
)

print(load_hsl_df_map().shape)
print(load_hsl_map_points().shape)
print(load_hsl_route_paths().shape)
print(load_hsl_route_options().shape)