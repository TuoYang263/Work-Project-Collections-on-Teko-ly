from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


STATES = [
    ("AC", "Acre", 12),
    ("AL", "Alagoas", 27),
    ("AP", "Amapá", 16),
    ("AM", "Amazonas", 13),
    ("BA", "Bahia", 29),
    ("CE", "Ceará", 23),
    ("DF", "Distrito Federal", 53),
    ("ES", "Espírito Santo", 32),
    ("GO", "Goiás", 52),
    ("MA", "Maranhão", 21),
    ("MT", "Mato Grosso", 51),
    ("MS", "Mato Grosso do Sul", 50),
    ("MG", "Minas Gerais", 31),
    ("PA", "Pará", 15),
    ("PB", "Paraíba", 25),
    ("PR", "Paraná", 41),
    ("PE", "Pernambuco", 26),
    ("PI", "Piauí", 22),
    ("RJ", "Rio de Janeiro", 33),
    ("RN", "Rio Grande do Norte", 24),
    ("RS", "Rio Grande do Sul", 43),
    ("RO", "Rondônia", 11),
    ("RR", "Roraima", 14),
    ("SC", "Santa Catarina", 42),
    ("SP", "São Paulo", 35),
    ("SE", "Sergipe", 28),
    ("TO", "Tocantins", 17),
]

OUTPUT = Path(
    "portal/public/geo/brazil-states.geojson"
)

features = []

for state_code, state_name, ibge_code in STATES:
    query = urlencode(
        {
            "formato": "application/vnd.geo+json",
            "qualidade": "minima",
        }
    )

    url = (
        "https://servicodados.ibge.gov.br/"
        f"api/v3/malhas/estados/{ibge_code}?{query}"
    )

    print(f"Downloading {state_code}...")

    request = Request(
        url,
        headers={
            "Accept": "application/vnd.geo+json",
            "User-Agent": "olist-analytics-portal/1.0",
        },
    )

    with urlopen(request, timeout=30) as response:
        raw_body = response.read()

    # IBGE may return gzip-compressed bytes.
    if raw_body.startswith(b"\x1f\x8b"):
        raw_body = gzip.decompress(raw_body)

    payload = json.loads(
        raw_body.decode("utf-8")
    )

    if payload.get("type") != "FeatureCollection":
        raise RuntimeError(
            f"{state_code}: expected FeatureCollection"
        )

    source_features = payload.get("features")

    if (
        not isinstance(source_features, list)
        or len(source_features) != 1
    ):
        raise RuntimeError(
            f"{state_code}: expected exactly one feature"
        )

    geometry = source_features[0].get("geometry")

    if geometry is None:
        raise RuntimeError(
            f"{state_code}: geometry missing"
        )

    features.append(
        {
            "type": "Feature",
            "id": state_code,
            "properties": {
                "state_code": state_code,
                "state_name": state_name,
            },
            "geometry": geometry,
        }
    )

if len(features) != 27:
    raise RuntimeError(
        f"Expected 27 states, got {len(features)}"
    )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ),
    encoding="utf-8",
)

print(f"Wrote {OUTPUT}")
print(f"features={len(features)}")
print(f"bytes={OUTPUT.stat().st_size}")
