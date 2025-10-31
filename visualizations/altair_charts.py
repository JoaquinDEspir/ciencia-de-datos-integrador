"""Interactive Altair visualisations for the real-estate dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

COUNTRIES_TOPOJSON_URL = (
    "https://raw.githubusercontent.com/vega/vega-datasets/master/data/world-110m.json"
)
ARGENTINA_ISO_NUMERIC = "032"


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
    """Create a drop-down parameter that defaults to mostrar todas las operaciones."""
    options = ["Todas"] + operations if operations else ["Todas"]
    return alt.param(
        name=name,
        value="Todas",
        bind=alt.binding_select(options=options, name=label),
    )


def _series_mode(series: pd.Series) -> str | None:
    mode = series.mode(dropna=True)
    return None if mode.empty else mode.iat[0]


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

    type_order = (
        df.groupby("tipo_propiedad")["precio_usd"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    operations = sorted(df["operacion"].dropna().unique().tolist())
    operation_param = _build_operation_param(
        operations, label="Operacion:", name="operacion_box"
    )

    base = (
        alt.Chart(df)
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
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("tipo_propiedad:N", legend=None),
        )
    )

    points = (
        base.mark_circle(size=40, opacity=0.35)
        .encode(
            x=alt.X("tipo_propiedad:N", sort=type_order),
            y=alt.Y("precio_usd:Q", scale=alt.Scale(type="log")),
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

    map_base = df.copy()
    map_base["lat_cell"] = map_base["prp_lat"].round(3)
    map_base["lng_cell"] = map_base["prp_lng"].round(3)

    aggregated = (
        map_base.groupby(["lat_cell", "lng_cell", "operacion"], as_index=False)
        .agg(
            median_price=("precio_usd", "median"),
            mean_price=("precio_usd", "mean"),
            max_price=("precio_usd", "max"),
            listings=("precio_usd", "size"),
            median_surface=("superficie_total", "median"),
        )
        .rename(columns={"lat_cell": "lat", "lng_cell": "lng"})
    )

    modal_info = (
        map_base.groupby(["lat_cell", "lng_cell", "operacion"])
        .agg(
            tipo_predominante=("tipo_propiedad", _series_mode),
            loc_predominante=("loc_desc", _series_mode),
        )
        .reset_index()
        .rename(columns={"lat_cell": "lat", "lng_cell": "lng"})
    )

    map_df = aggregated.merge(modal_info, on=["lat", "lng", "operacion"], how="left")
    map_df = map_df[map_df["median_price"].notna() & (map_df["median_price"] > 0)]
    if map_df.empty:
        raise ValueError(
            "No hay registros suficientes para construir el mapa de precios agrupado."
        )

    operations = sorted(map_df["operacion"].dropna().unique().tolist())
    operation_param = _build_operation_param(
        operations, label="Operacion:", name="operacion_mapa"
    )

    background = (
        alt.Chart(alt.topo_feature(COUNTRIES_TOPOJSON_URL, "countries"))
        .transform_filter(f"datum.id == '{ARGENTINA_ISO_NUMERIC}'")
        .mark_geoshape(fill="#f5f5f5", stroke="#bdbdbd")
        .project(type="mercator", scale=2000, center=[-68.85, -32.89])
    )

    points = (
        alt.Chart(map_df)
        .add_params(operation_param)
        .transform_filter(
            (operation_param == "Todas")
            | (alt.datum.operacion == operation_param)
        )
        .mark_circle(opacity=0.85, stroke="#1a1a1a", strokeWidth=0.35)
        .encode(
            longitude=alt.Longitude("lng:Q"),
            latitude=alt.Latitude("lat:Q"),
            size=alt.Size(
                "listings:Q",
                title="Cantidad de avisos",
                scale=alt.Scale(type="sqrt", range=[25, 550]),
            ),
            color=alt.Color(
                "median_price:Q",
                title="Mediana precio (USD)",
                scale=alt.Scale(type="log", scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("operacion:N", title="Operacion"),
                alt.Tooltip("tipo_predominante:N", title="Tipo predominante"),
                alt.Tooltip("listings:Q", title="Cantidad de avisos"),
                alt.Tooltip("median_price:Q", title="Mediana precio USD", format=",.0f"),
                alt.Tooltip("mean_price:Q", title="Precio promedio USD", format=",.0f"),
                alt.Tooltip("max_price:Q", title="Maximo USD", format=",.0f"),
                alt.Tooltip("loc_predominante:N", title="Localidad predominante"),
                alt.Tooltip(
                    "median_surface:Q", title="Mediana sup. total (m2)", format=",.0f"
                ),
            ],
        )
    )

    return (
        (background + points)
        .properties(
            title="Concentracion de precio listado (USD) en el Gran Mendoza",
            width=700,
            height=500,
        )
        .resolve_scale(size="independent", color="independent")
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
