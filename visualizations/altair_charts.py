"""Interactive Altair visualisations for the real-estate dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import altair as alt
except ImportError as exc:  # pragma: no cover - only hits when Altair is missing.
    alt = None  # type: ignore[assignment]
    _ALT_IMPORT_ERROR = exc
else:  # pragma: no cover - simple configuration hook.
    alt.data_transformers.disable_max_rows()
    _ALT_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "include" / "data" / "processed" / "propiedades_clean.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualizations" / "charts"
MENDOZA_BOUNDARY_PATH = (
    PROJECT_ROOT / "include" / "data" / "geo" / "mendoza_boundary.geojson"
)

BOXPLOT_PRICE_DOMAIN = (10_000, 1_000_000)
MAP_BIN_STEP_DEGREES = 0.01
MAP_PRICE_DOMAIN = (10_000, 1_000_000)
MIN_LISTINGS_PER_CELL = 2


@dataclass(frozen=True)
class PreparedData:
    """Container for the cleaned dataframe and common filtered subsets."""

    full: pd.DataFrame
    with_price_usd: pd.DataFrame
    with_geo: pd.DataFrame


def _require_altair() -> None:
    if alt is None:  # pragma: no cover - depends on optional dependency.
        message = (
            "Altair no esta instalado. Ejecuta `pip install altair` antes de generar "
            "las visualizaciones interactivas."
        )
        raise RuntimeError(message) from _ALT_IMPORT_ERROR


def _build_operation_param(
    operations: List[str], *, label: str, name: str
):  # pragma: no cover - thin wrapper.
    """Create a drop-down parameter that defaults a mostrar todas las operaciones."""
    options = ["Todas"] + operations if operations else ["Todas"]
    return alt.param(
        name=name,
        value="Todas",
        bind=alt.binding_select(options=options, name=label),
    )


def _series_mode(series: pd.Series) -> str | None:
    mode = series.mode(dropna=True)
    return None if mode.empty else mode.iat[0]


@lru_cache(maxsize=1)
def _mendoza_boundary() -> Dict[str, object]:
    """Load boundary polygon for Mendoza province as a FeatureCollection."""
    if not MENDOZA_BOUNDARY_PATH.exists():
        raise RuntimeError(
            "No se encuentra el poligono de Mendoza. Ejecuta el script de descarga."
        )
    geometry = json.loads(MENDOZA_BOUNDARY_PATH.read_text(encoding="utf-8"))
    if geometry.get("type") != "FeatureCollection":
        geometry = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": geometry}
            ],
        }
    return geometry


def _ring_contains_point(lon: float, lat: float, ring: List[List[float]]) -> bool:
    """Ray-casting algorithm to determine if a point is inside a linear ring."""
    inside = False
    if len(ring) < 3:
        return False
    points = ring[:-1] if ring[0] == ring[-1] else ring
    if len(points) < 3:
        return False
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        intersects = ((y1 > lat) != (y2 > lat)) and (
            lon
            < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12)  # avoid division by zero
            + x1
        )
        if intersects:
            inside = not inside
    return inside


def _polygon_contains_point(
    lon: float, lat: float, coordinates: List[List[List[float]]]
) -> bool:
    """Return True if point is inside polygon defined by coordinates."""
    if not coordinates:
        return False
    if not _ring_contains_point(lon, lat, coordinates[0]):
        return False
    for hole in coordinates[1:]:
        if _ring_contains_point(lon, lat, hole):
            return False
    return True


def _geometry_contains_point(lon: float, lat: float, geometry: Dict[str, object]) -> bool:
    """Evaluate whether a point is inside (Multi)Polygon geometry."""
    gtype = geometry.get("type")
    if gtype == "Polygon":
        return _polygon_contains_point(lon, lat, geometry.get("coordinates", []))
    if gtype == "MultiPolygon":
        for polygon in geometry.get("coordinates", []):
            if _polygon_contains_point(lon, lat, polygon):
                return True
        return False
    return False


def load_properties_data(csv_path: Path = DATA_PATH) -> PreparedData:
    """Read and tidy the properties dataset to feed Altair charts."""
    df = pd.read_csv(csv_path, sep=";", decimal=".")

    numeric_columns = [
        "prp_pre_dol",
        "prp_pre",
        "banos",
        "dormitorios",
        "cochera",
        "cocheras",
        "sup_total",
        "sup_cubierta",
        "prp_lat",
        "prp_lng",
        "fotos_cantidad",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["prp_alta"] = pd.to_datetime(df["prp_alta"], errors="coerce")
    df["prp_mod"] = pd.to_datetime(df["prp_mod"], errors="coerce")

    df["operacion"] = df["con_desc"].fillna("Desconocida").str.title()
    df["tipo_propiedad"] = (
        df["grupo_tip_desc"].fillna(df["tip_desc"]).fillna("Sin Titulo")
    )
    df["tipo_propiedad"] = df["tipo_propiedad"].str.replace("_", " ").str.title()

    df["precio_usd"] = df["prp_pre_dol"].where(df["prp_pre_dol"] > 0)
    df["precio_ars"] = df["prp_pre"].where(df["prp_pre"] > 0)
    df["superficie_total"] = df["sup_total"].where(df["sup_total"] > 0)
    df["superficie_cubierta"] = df["sup_cubierta"].where(df["sup_cubierta"] > 0)

    df["cocheras_total"] = df[["cocheras", "cochera"]].bfill(axis=1).iloc[:, 0]
    df["cocheras_total"] = df["cocheras_total"].where(df["cocheras_total"] >= 0)

    with_price_usd = df[df["precio_usd"].notna()].copy()
    with_geo = with_price_usd[
        with_price_usd["prp_lat"].notna() & with_price_usd["prp_lng"].notna()
    ].copy()

    return PreparedData(full=df, with_price_usd=with_price_usd, with_geo=with_geo)


def price_distribution_chart(data: PreparedData) -> alt.Chart:
    """Box plot of USD prices by property type and operation."""
    _require_altair()
    df = data.with_price_usd.copy()
    if df.empty:
        raise ValueError("No hay registros con precio en USD para la visualizacion.")

    df = df[df["tipo_propiedad"].notna()]
    floor, ceiling = BOXPLOT_PRICE_DOMAIN
    trimmed = df[df["precio_usd"].between(floor, ceiling)]
    if trimmed.empty:
        trimmed = df[df["precio_usd"].notna()]

    type_order = (
        trimmed.groupby("tipo_propiedad")["precio_usd"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    operations = sorted(trimmed["operacion"].dropna().unique().tolist())
    operation_param = _build_operation_param(
        operations, label="Operacion:", name="operacion_box"
    )

    base = (
        alt.Chart(trimmed)
        .add_params(operation_param)
        .transform_filter(
            (operation_param == "Todas")
            | (alt.datum.operacion == operation_param)
        )
        .properties(
            title="Distribucion de precios listados (USD) por tipo de propiedad"
        )
    )

    box = (
        base.mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(
                "tipo_propiedad:N",
                sort=type_order,
                title="Tipo de propiedad",
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y(
                "precio_usd:Q",
                title="Precio listado (USD)",
                scale=alt.Scale(
                    type="log", domain=BOXPLOT_PRICE_DOMAIN, clamp=True
                ),
            ),
            color=alt.Color("tipo_propiedad:N", legend=None),
        )
    )

    points = (
        base.mark_circle(size=40, opacity=0.35)
        .encode(
            x=alt.X("tipo_propiedad:N", sort=type_order),
            y=alt.Y(
                "precio_usd:Q",
                scale=alt.Scale(
                    type="log", domain=BOXPLOT_PRICE_DOMAIN, clamp=True
                ),
            ),
            color=alt.Color("tipo_propiedad:N", legend=None),
            tooltip=[
                alt.Tooltip("tip_desc:N", title="Tipo"),
                alt.Tooltip("operacion:N", title="Operacion"),
                alt.Tooltip("loc_desc:N", title="Localidad"),
                alt.Tooltip("precio_usd:Q", title="Precio USD", format=",.0f"),
                alt.Tooltip(
                    "superficie_total:Q", title="Sup. total (m2)", format=",.0f"
                ),
            ],
        )
    )

    return (box + points).properties(width=700, height=400)


def price_vs_surface_chart(data: PreparedData) -> alt.Chart:
    """Scatter plot comparing total surface against USD price."""
    _require_altair()
    df = data.with_price_usd.copy()
    df = df[df["superficie_total"].notna()]
    if df.empty:
        raise ValueError(
            "No hay registros con superficie total valida para la visualizacion."
        )

    operations = sorted(df["operacion"].dropna().unique().tolist())
    operation_param = _build_operation_param(
        operations, label="Operacion:", name="operacion_surface"
    )

    type_legend_selection = alt.selection_multi(
        fields=["tipo_propiedad"], bind="legend"
    )

    chart = (
        alt.Chart(df)
        .add_params(operation_param)
        .add_selection(type_legend_selection)
        .transform_filter(
            (operation_param == "Todas")
            | (alt.datum.operacion == operation_param)
        )
        .transform_filter(type_legend_selection)
        .mark_circle(stroke="#212121", strokeWidth=0.4, opacity=0.65)
        .encode(
            x=alt.X(
                "superficie_total:Q",
                title="Superficie total (m2)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", tickCount=6),
            ),
            y=alt.Y(
                "precio_usd:Q",
                title="Precio listado (USD)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", tickCount=6),
            ),
            color=alt.Color(
                "tipo_propiedad:N",
                legend=alt.Legend(title="Tipo de propiedad"),
                scale=alt.Scale(scheme="tableau20"),
            ),
            size=alt.Size(
                "dormitorios:Q",
                title="Dormitorios",
                scale=alt.Scale(type="sqrt", range=[25, 320]),
            ),
            tooltip=[
                alt.Tooltip("tip_desc:N", title="Tipo"),
                alt.Tooltip("operacion:N", title="Operacion"),
                alt.Tooltip("loc_desc:N", title="Localidad"),
                alt.Tooltip("precio_usd:Q", title="Precio USD", format=",.0f"),
                alt.Tooltip(
                    "superficie_total:Q", title="Sup. total (m2)", format=",.0f"
                ),
                alt.Tooltip("dormitorios:Q", title="Dormitorios"),
                alt.Tooltip("banos:Q", title="Banos"),
            ],
        )
        .properties(
            title="Relacion entre metros totales y precio listado",
            width=700,
            height=400,
        )
    )

    return chart


def price_density_map(data: PreparedData) -> alt.Chart:
    """Map of Mendoza showing price intensity by latitude and longitude."""
    _require_altair()
    df = data.with_geo.copy()
    if df.empty:
        raise ValueError(
            "No hay registros georreferenciados con precio en USD para el mapa."
        )

    df = df[
        df["prp_lat"].between(-34.5, -31.5) & df["prp_lng"].between(-70.0, -67.0)
    ]
    if df.empty:
        raise ValueError("No hay registros dentro del recorte geografico definido.")

    bin_step = MAP_BIN_STEP_DEGREES
    df["lng_bin"] = np.floor(df["prp_lng"] / bin_step) * bin_step
    df["lat_bin"] = np.floor(df["prp_lat"] / bin_step) * bin_step
    grouped = (
        df.groupby(["operacion", "lng_bin", "lat_bin"], as_index=False)
        .agg(
            median_price=("precio_usd", "median"),
            mean_price=("precio_usd", "mean"),
            max_price=("precio_usd", "max"),
            listings=("precio_usd", "size"),
            median_surface=("superficie_total", "median"),
            tipo_predominante=("tipo_propiedad", _series_mode),
            loc_predominante=("loc_desc", _series_mode),
        )
        .dropna(subset=["median_price"])
    )
    grouped = grouped[grouped["median_price"] > 0]
    grouped = grouped[grouped["listings"] >= MIN_LISTINGS_PER_CELL]
    if grouped.empty:
        raise ValueError(
            "No hay registros suficientes para construir el mapa de precios agrupado."
        )

    grouped["lng_bin_end"] = grouped["lng_bin"] + bin_step
    grouped["lat_bin_end"] = grouped["lat_bin"] + bin_step

    boundary_geo = _mendoza_boundary()
    boundary_features = boundary_geo["features"]

    features = []
    for row in grouped.itertuples():
        lon_center = (row.lng_bin + row.lng_bin_end) / 2
        lat_center = (row.lat_bin + row.lat_bin_end) / 2
        if not any(
            _geometry_contains_point(lon_center, lat_center, feature["geometry"])
            for feature in boundary_features
        ):
            continue
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [row.lng_bin, row.lat_bin],
                    [row.lng_bin_end, row.lat_bin],
                    [row.lng_bin_end, row.lat_bin_end],
                    [row.lng_bin, row.lat_bin_end],
                    [row.lng_bin, row.lat_bin],
                ]
            ],
        }
        properties = {
            "operacion": row.operacion,
            "median_price": float(row.median_price) if pd.notna(row.median_price) else None,
            "mean_price": float(row.mean_price) if pd.notna(row.mean_price) else None,
            "max_price": float(row.max_price) if pd.notna(row.max_price) else None,
            "listings": int(row.listings),
            "median_surface": float(row.median_surface)
            if pd.notna(row.median_surface)
            else None,
            "tipo_predominante": row.tipo_predominante,
            "loc_predominante": row.loc_predominante,
        }
        features.append(
            {"type": "Feature", "properties": properties, "geometry": geometry}
        )

    if not features:
        raise ValueError(
            "No hay celdas suficientes para visualizar la concentracion de precios."
        )

    operations = sorted(
        {
            feature["properties"]["operacion"]
            for feature in features
            if feature["properties"]["operacion"]
        }
    )
    operation_param = _build_operation_param(
        operations, label="Operacion:", name="operacion_mapa"
    )

    boundary_data = alt.Data(values=boundary_geo["features"])
    heatmap_data = alt.Data(values=features)

    price_values = [
        feature["properties"]["median_price"]
        for feature in features
        if feature["properties"]["median_price"] is not None
    ]
    if price_values:
        lower = max(MAP_PRICE_DOMAIN[0], min(price_values))
        upper = min(MAP_PRICE_DOMAIN[1], max(price_values))
        if lower >= upper:
            upper = max(lower * 1.1, lower + 1)
        price_domain = [lower, upper]
    else:
        price_domain = list(MAP_PRICE_DOMAIN)

    listings_values = [
        feature["properties"]["listings"]
        for feature in features
        if feature["properties"]["listings"] is not None
    ]
    max_listings = max(listings_values) if listings_values else MIN_LISTINGS_PER_CELL

    boundary = (
        alt.Chart(boundary_data)
        .mark_geoshape(fill=None, stroke="#60708d", strokeWidth=0.8)
        .project(type="mercator", fit=boundary_geo)
    )

    heatmap = (
        alt.Chart(heatmap_data)
        .add_params(operation_param)
        .transform_filter(
            (operation_param == "Todas")
            | (alt.datum["properties"]["operacion"] == operation_param)
        )
        .mark_geoshape(stroke="#0b1120", strokeWidth=0.15)
        .encode(
            color=alt.Color(
                "properties.median_price:Q",
                title="Mediana precio (USD)",
                scale=alt.Scale(
                    type="log",
                    domain=price_domain,
                    clamp=True,
                    scheme="viridis",
                ),
            ),
            opacity=alt.Opacity(
                "properties.listings:Q",
                title="Cantidad de avisos",
                scale=alt.Scale(
                    domain=[MIN_LISTINGS_PER_CELL, max_listings],
                    range=[0.35, 0.95],
                    clamp=True,
                ),
            ),
            tooltip=[
                alt.Tooltip("properties.operacion:N", title="Operacion"),
                alt.Tooltip("properties.tipo_predominante:N", title="Tipo predominante"),
                alt.Tooltip("properties.loc_predominante:N", title="Localidad predominante"),
                alt.Tooltip("properties.listings:Q", title="Cantidad de avisos"),
                alt.Tooltip(
                    "properties.median_price:Q",
                    title="Mediana precio USD",
                    format=",.0f",
                ),
                alt.Tooltip(
                    "properties.mean_price:Q",
                    title="Precio promedio USD",
                    format=",.0f",
                ),
                alt.Tooltip(
                    "properties.max_price:Q",
                    title="Maximo USD",
                    format=",.0f",
                ),
                alt.Tooltip(
                    "properties.median_surface:Q",
                    title="Mediana sup. total (m2)",
                    format=",.0f",
                ),
            ],
        )
        .project(type="mercator", fit=boundary_geo)
        .properties(width=700, height=500)
    )

    return (
        (boundary + heatmap)
        .properties(title="Concentracion de precio listado (USD) en el Gran Mendoza")
        .resolve_scale(color="independent")
    )


def save_charts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Path]:
    """Generate all charts and persist them as HTML files."""
    _require_altair()
    data = load_properties_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = {
        "price_distribution.html": price_distribution_chart(data),
        "price_vs_surface.html": price_vs_surface_chart(data),
        "price_map.html": price_density_map(data),
    }

    saved_paths: Dict[str, Path] = {}
    for filename, chart in charts.items():
        chart_path = output_dir / filename
        chart.save(chart_path)
        saved_paths[filename] = chart_path
    return saved_paths


if __name__ == "__main__":
    paths = save_charts()
    for name, path in paths.items():
        print(f"Chart '{name}' guardado en: {path}")
