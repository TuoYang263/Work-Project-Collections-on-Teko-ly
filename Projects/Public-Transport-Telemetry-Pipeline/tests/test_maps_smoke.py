from pathlib import Path
from streamlit_app.utils.maps import build_map_bundle

MAP_DATA_DIR = Path("data/gold/hsl")


def test_build_map_bundle_smoke():
    bundle = build_map_bundle(MAP_DATA_DIR)

    assert isinstance(bundle, dict)
    assert "routes" in bundle
    assert "points" in bundle
    assert "paths" in bundle
    assert "center" in bundle
    assert "raw" in bundle

    assert "lat" in bundle["center"]
    assert "lon" in bundle["center"]