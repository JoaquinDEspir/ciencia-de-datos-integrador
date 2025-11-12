"""Streamlit app to explore Mendoza real-estate data and test the price model."""

from __future__ import annotations

from typing import Dict, List, Tuple

import math

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
except ImportError as exc:  # pragma: no cover - optional dependency guard
    st_folium = None  # type: ignore[assignment]
    _ST_FOLIUM_IMPORT_ERROR = exc
else:
    _ST_FOLIUM_IMPORT_ERROR = None

from app.model import GEO_ZONE_COLUMN, ModelArtifacts, TARGET_COLUMN, train_price_model
from app.geography import infer_neighborhood, list_neighborhoods
from visualizations.altair_charts import (
    PRICE_COMPARISON_DEFAULT,
    PRICE_COMPARISON_FIELDS,
    PreparedData,
    load_properties_data,
    price_density_geojson,
    price_distribution_chart,
    price_vs_surface_chart,
)
DEFAULT_LAT = -32.889
DEFAULT_LNG = -68.845
EARTH_RADIUS_KM = 6371.0
COMPARABLE_DISTANCE_WINDOWS_KM = (0.8, 1.5, 3.0, 6.0)
SURFACE_TOLERANCES = (0.25, 0.4, 0.6)
MIN_COMPARABLES = 4
MAX_COMPARABLES = 12


def _ensure_streamlit_folium() -> bool:
    """Check if streamlit-folium is available and inform the user otherwise."""
    if st_folium is not None:
        return True
    message = (
        "La funcionalidad de mapa requiere instalar `streamlit-folium`. "
        "Asegurate de ejecutar `pip install -r requirements.txt` y "
        "redeployar la app."
    )
    st.error(message)
    if _ST_FOLIUM_IMPORT_ERROR:
        st.caption(f"Detalle: {_ST_FOLIUM_IMPORT_ERROR}")
    return False


def _geo_distance_km(
    lat: float, lng: float, latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)
    lat2 = np.radians(latitudes)
    lng2 = np.radians(longitudes)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _filter_by_surface_window(df: pd.DataFrame, superficie: float) -> pd.DataFrame:
    """Restrict candidates to a surface window, relaxing bounds if needed."""
    if df.empty or superficie is None or superficie <= 0:
        return df
    for tolerance in SURFACE_TOLERANCES:
        lower = max(10.0, superficie * (1 - tolerance))
        upper = superficie * (1 + tolerance)
        window = df[df["superficie_total"].between(lower, upper)]
        if not window.empty:
            return window
    return df


def find_comparable_properties(
    prepared: PreparedData,
    *,
    lat: float,
    lng: float,
    operacion: str,
    tipo: str,
    superficie: float,
    zona: str,
    localidad: str,
    max_results: int = MAX_COMPARABLES,
    min_results: int = MIN_COMPARABLES,
) -> pd.DataFrame:
    """Return nearby dataset listings comparable to the user input."""
    geo_df = prepared.with_geo
    if geo_df.empty:
        return pd.DataFrame()

    candidates = geo_df[
        geo_df["superficie_total"].notna() & (geo_df["superficie_total"] > 0)
    ].copy()
    if candidates.empty:
        return candidates

    if "precio_m2" not in candidates.columns:
        candidates["precio_m2"] = (
            candidates[TARGET_COLUMN] / candidates["superficie_total"]
        )
        candidates.loc[
            candidates["superficie_total"] <= 0, "precio_m2"
        ] = np.nan
    if "precio_m2_cubierto" not in candidates.columns:
        candidates["precio_m2_cubierto"] = (
            candidates[TARGET_COLUMN] / candidates["superficie_cubierta"]
        )
        candidates.loc[
            candidates["superficie_cubierta"] <= 0, "precio_m2_cubierto"
        ] = np.nan

    base_subset = candidates[
        (candidates["operacion"] == operacion) & (candidates["tipo_propiedad"] == tipo)
    ]
    if base_subset.empty:
        base_subset = candidates[candidates["operacion"] == operacion]
    if base_subset.empty:
        base_subset = candidates

    zone_subset = (
        base_subset[base_subset[GEO_ZONE_COLUMN] == zona]
        if zona
        else pd.DataFrame()
    )
    loc_subset = (
        base_subset[base_subset["loc_desc"] == localidad]
        if localidad
        else pd.DataFrame()
    )

    working = (
        zone_subset
        if not zone_subset.empty
        else loc_subset
        if not loc_subset.empty
        else base_subset
    )
    working = _filter_by_surface_window(working, superficie)
    if working.empty:
        working = _filter_by_surface_window(base_subset, superficie)

    if working.empty:
        return working

    distances = _geo_distance_km(
        lat,
        lng,
        working["prp_lat"].to_numpy(),
        working["prp_lng"].to_numpy(),
    )
    working = working.assign(distance_km=distances)

    selected = pd.DataFrame()
    for radius in (*COMPARABLE_DISTANCE_WINDOWS_KM, None):
        subset = working if radius is None else working[working["distance_km"] <= radius]
        if subset.empty:
            continue
        selected = subset
        if radius is None or len(subset) >= min_results:
            break

    if selected.empty:
        selected = working

    return selected.sort_values("distance_km").head(max_results)


@st.cache_data(show_spinner=False)
def get_prepared_data() -> PreparedData:
    return load_properties_data()


@st.cache_resource(show_spinner=False)
def get_model_artifacts() -> ModelArtifacts:
    prepared = get_prepared_data()
    return train_price_model(prepared.with_price_usd.copy())


def build_filtered_data(
    prepared: PreparedData,
    operaciones: List[str],
    tipos: List[str],
) -> PreparedData:
    df = prepared.full.copy()
    if operaciones:
        df = df[df["operacion"].isin(operaciones)]
    if tipos:
        df = df[df["tipo_propiedad"].isin(tipos)]
    with_price = df[df[TARGET_COLUMN].notna()]
    with_geo = with_price[
        with_price["prp_lat"].notna() & with_price["prp_lng"].notna()
    ]
    return PreparedData(
        full=df,
        with_price_usd=with_price,
        with_geo=with_geo,
        barrio_centroids=prepared.barrio_centroids,
        neighborhoods_by_loc=prepared.neighborhoods_by_loc,
    )


def render_price_density_map(filtered: PreparedData) -> None:
    if not _ensure_streamlit_folium():
        return
    from branca.colormap import LinearColormap
    import folium

    try:
        feature_collection, price_domain, max_listings, _boundary, _ops = (
            price_density_geojson(filtered)
        )
    except (RuntimeError, ValueError) as exc:
        st.warning(str(exc))
        return

    features = feature_collection["features"]
    if not features:
        st.warning("Sin celdas suficientes para construir el mapa de calor.")
        return

    lons: List[float] = []
    lats: List[float] = []
    for feature in features:
        coords = feature["geometry"]["coordinates"][0]
        for lon, lat in coords:
            lons.append(lon)
            lats.append(lat)
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    base_map = folium.Map(
        location=[center_lat, center_lon],
        tiles="CartoDB positron",
        zoom_start=11,
        control_scale=True,
    )

    valid_prices = [
        feature["properties"]["median_price"]
        for feature in features
        if feature["properties"].get("median_price")
    ]
    if not valid_prices:
        st.warning("No hay precios suficientes para graficar el mapa.")
        return

    lower = max(min(valid_prices), price_domain[0])
    upper = max(max(valid_prices), lower * 1.05)
    log_min = math.log10(lower)
    log_max = math.log10(upper)
    colormap = LinearColormap(
        colors=["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=log_min,
        vmax=log_max,
        caption="Mediana precio (USD) – escala log",
    )

    def style_function(feature: Dict[str, object]) -> Dict[str, object]:
        properties = feature["properties"]
        price = properties.get("median_price")
        color = "#9e9e9e"
        if isinstance(price, (int, float)) and price > 0:
            color = colormap(math.log10(price))
        listings = properties.get("listings") or 0
        opacity = 0.25 + 0.7 * min(listings / max(max_listings, 1), 1.0)
        return {
            "fillColor": color,
            "color": "#4b5563",
            "weight": 0.4,
            "fillOpacity": opacity,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=[
            "operacion",
            "tipo_predominante",
            "loc_predominante",
            "listings",
            "median_price",
            "mean_price",
            "max_price",
            "median_surface",
        ],
        aliases=[
            "Operacion",
            "Tipo predominante",
            "Localidad predominante",
            "Cantidad de avisos",
            "Mediana precio USD",
            "Precio promedio USD",
            "Maximo USD",
            "Mediana sup. total (m2)",
        ],
        localize=True,
        sticky=False,
        labels=True,
    )

    folium.GeoJson(
        feature_collection,
        name="Mediana precio (USD)",
        style_function=style_function,
        tooltip=tooltip,
        highlight_function=lambda _: {"weight": 1.1, "color": "#111827"},
    ).add_to(base_map)
    colormap.add_to(base_map)

    st_folium(base_map, width=900, height=520, returned_objects=[])


def render_visualisations(filtered: PreparedData) -> None:
    try:
        st.altair_chart(price_distribution_chart(filtered), use_container_width=True)
    except (RuntimeError, ValueError) as exc:
        st.warning(str(exc))

    metric_options = list(PRICE_COMPARISON_FIELDS.keys())
    default_index = (
        metric_options.index(PRICE_COMPARISON_DEFAULT)
        if PRICE_COMPARISON_DEFAULT in PRICE_COMPARISON_FIELDS
        else 0
    )
    selected_metric = st.selectbox(
        "Variable para comparar con el precio:",
        options=metric_options,
        index=default_index,
        format_func=lambda key: str(PRICE_COMPARISON_FIELDS[key]["label"]),
    )

    try:
        st.altair_chart(
            price_vs_surface_chart(filtered, metric=selected_metric),
            use_container_width=True,
        )
    except (RuntimeError, ValueError) as exc:
        st.warning(str(exc))
    render_price_density_map(filtered)


def render_sidebar(artifacts: ModelArtifacts) -> Tuple[List[str], List[str]]:
    st.sidebar.header("Filtros")
    operaciones = st.sidebar.multiselect(
        "Operacion",
        options=artifacts.feature_options["operacion"],
        default=artifacts.feature_options["operacion"],
    )
    tipos = st.sidebar.multiselect(
        "Tipo de propiedad",
        options=artifacts.feature_options["tipo_propiedad"],
        default=artifacts.feature_options["tipo_propiedad"][:5],
        help="Seleccion inicial de los tipos mas frecuentes.",
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Selecciones arriba para refinar datos.")
    return operaciones, tipos


def render_summary(prepared: PreparedData, artifacts: ModelArtifacts) -> None:
    df = prepared.with_price_usd
    total_listings = len(df)
    avg_price = float(df[TARGET_COLUMN].mean())
    avg_surface = float(df["superficie_total"].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("Avisos con precio USD", f"{total_listings:,}")
    col2.metric("Precio promedio USD", f"{avg_price:,.0f}")
    col3.metric("Superficie promedio m2", f"{avg_surface:,.0f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("MAE validacion USD", f"{artifacts.metrics['mae']:,.0f}")
    col5.metric("RMSE validacion USD", f"{artifacts.metrics['rmse']:,.0f}")
    col6.metric("R2 validacion", f"{artifacts.metrics['r2']:.2f}")

    st.subheader("Distribucion de precios (USD)")
    summary = artifacts.price_summary
    st.write(
        f"Min: {summary['min']:,.0f} | Mediana: {summary['p50']:,.0f} | "
        f"P90: {summary['p90']:,.0f} | Max: {summary['max']:,.0f}"
    )
    st.dataframe(
        df[["tipo_propiedad", "operacion", "loc_desc", TARGET_COLUMN, "superficie_total"]]
        .sort_values(TARGET_COLUMN, ascending=False)
        .head(50),
        use_container_width=True,
    )


def render_prediction_ui(prepared: PreparedData, artifacts: ModelArtifacts) -> None:
    st.subheader("Ingresar una propiedad para estimar precio in situ")

    col1, col2, col3 = st.columns(3)
    operacion = col1.selectbox(
        "Operacion",
        options=artifacts.feature_options["operacion"],
    )
    tipo = col2.selectbox(
        "Tipo de propiedad",
        options=artifacts.feature_options["tipo_propiedad"],
    )
    localidad = col3.selectbox(
        "Localidad",
        options=artifacts.feature_options["loc_desc"],
    )

    loc_key = str(localidad).strip()
    loc_norm = loc_key.lower()
    reference_barrios = list_neighborhoods(loc_key)
    dataset_barrios = list(prepared.neighborhoods_by_loc.get(loc_norm, []))
    available_barrios = reference_barrios or []
    for label in dataset_barrios:
        if label not in available_barrios:
            available_barrios.append(label)
    if not available_barrios:
        available_barrios = [
            f"{loc_key} - Otros",
            f"{loc_key} - Sin Coordenadas",
        ]

    last_loc = st.session_state.get("last_localidad")
    if last_loc != loc_norm:
        st.session_state["last_localidad"] = loc_norm
        st.session_state["barrio_select"] = available_barrios[0]

    if st.button("Actualizar barrios", type="secondary"):
        st.session_state["last_localidad"] = loc_norm
        st.session_state["barrio_select"] = available_barrios[0]

    current_barrio = st.session_state.get("barrio_select", available_barrios[0])
    if current_barrio not in available_barrios:
        current_barrio = available_barrios[0]
        st.session_state["barrio_select"] = current_barrio

    barrio = st.selectbox(
        "Barrio / zona",
        available_barrios,
        key="barrio_select",
    )

    col4, col5 = st.columns(2)
    superficie = col4.number_input(
        "Superficie total (m2)",
        min_value=10.0,
        max_value=5000.0,
        value=120.0,
        step=10.0,
    )
    superficie_cubierta = col5.number_input(
        "Superficie cubierta (m2)",
        min_value=0.0,
        max_value=5000.0,
        value=float(superficie),
        step=10.0,
    )

    col6, col7, col8 = st.columns(3)
    dormitorios = col6.slider("Dormitorios", min_value=0, max_value=10, value=2)
    banos = col7.slider("Banos", min_value=1, max_value=8, value=2)
    tiene_cochera = col8.checkbox("Tiene cochera", value=True)
    cocheras = 1 if tiene_cochera else 0

    default_lat, default_lng = prepared.barrio_centroids.get(
        barrio, (None, None)
    )
    lat_value = float(default_lat) if default_lat is not None else DEFAULT_LAT
    lng_value = float(default_lng) if default_lng is not None else DEFAULT_LNG

    coord_key = f"coords_{loc_norm}_{barrio}"
    lat_input_key = f"lat_input_{coord_key}"
    lng_input_key = f"lng_input_{coord_key}"
    if coord_key not in st.session_state:
        st.session_state[coord_key] = {"lat": lat_value, "lng": lng_value}
        st.session_state[lat_input_key] = lat_value
        st.session_state[lng_input_key] = lng_value

    show_map_key = f"show_map_{loc_norm}_{barrio}"
    map_btn_col, _ = st.columns([1, 3])
    if map_btn_col.button("Seleccionar ubicacion en mapa", key=f"map_btn_{loc_norm}_{barrio}"):
        st.session_state[show_map_key] = True

    if st.session_state.get(show_map_key):
        if not _ensure_streamlit_folium():
            st.session_state[show_map_key] = False
            st.info("No se puede mostrar el mapa sin `streamlit-folium` instalado.")
        else:
            import folium

            map_container = st.container()
            with map_container:
                st.markdown("Haz click en el mapa para fijar la ubicacion.")
                map_obj = folium.Map(
                    location=[
                        st.session_state[coord_key]["lat"],
                        st.session_state[coord_key]["lng"],
                    ],
                    zoom_start=14,
                    control_scale=True,
                )
                folium.Marker(
                    [
                        st.session_state[coord_key]["lat"],
                        st.session_state[coord_key]["lng"],
                    ],
                    tooltip="Posicion seleccionada",
                ).add_to(map_obj)
                map_obj.add_child(folium.LatLngPopup())
                map_data = st_folium(
                    map_obj,
                    height=360,
                    width=None,
                    key=f"map_{loc_norm}_{barrio}",
                )
                if map_data and map_data.get("last_clicked"):
                    st.session_state[coord_key]["lat"] = map_data["last_clicked"]["lat"]
                    st.session_state[coord_key]["lng"] = map_data["last_clicked"]["lng"]
                    st.session_state[lat_input_key] = st.session_state[coord_key]["lat"]
                    st.session_state[lng_input_key] = st.session_state[coord_key]["lng"]

            if st.button("Cerrar mapa", key=f"close_map_{loc_norm}_{barrio}"):
                st.session_state[show_map_key] = False

    col9, col10 = st.columns(2)
    st.session_state[coord_key]["lat"] = col9.number_input(
        "Latitud",
        min_value=-34.5,
        max_value=-31.0,
        value=float(st.session_state[lat_input_key]),
        step=0.001,
        format="%.4f",
        key=lat_input_key,
    )
    st.session_state[coord_key]["lng"] = col10.number_input(
        "Longitud",
        min_value=-70.5,
        max_value=-66.5,
        value=float(st.session_state[lng_input_key]),
        step=0.001,
        format="%.4f",
        key=lng_input_key,
    )

    latitud = float(st.session_state[coord_key]["lat"])
    longitud = float(st.session_state[coord_key]["lng"])

    inferred_zone = infer_neighborhood(loc_key, float(latitud), float(longitud))
    if inferred_zone != barrio:
        st.caption(
            f"Zona seleccionada: {barrio} | Coordenadas sugieren: {inferred_zone}"
        )
    else:
        st.caption(
            f"Zona seleccionada: {barrio} (lat {latitud:.4f}, lng {longitud:.4f})"
        )

    comparables_df = find_comparable_properties(
        prepared,
        lat=latitud,
        lng=longitud,
        operacion=operacion,
        tipo=tipo,
        superficie=float(superficie),
        zona=barrio,
        localidad=localidad,
    )

    col_valor_m2, col_valor_m2_cub, col_valor_detalle = st.columns([1, 1, 2])
    if comparables_df.empty:
        col_valor_m2.info("Sin referencias cercanas.")
        col_valor_m2_cub.info("Sin valores cubiertos.")
        col_valor_detalle.caption(
            "Ajusta coordenadas, operacion o tipo para ver precios por m2."
        )
    else:
        comparables_len = len(comparables_df)
        price_series = comparables_df["precio_m2"].dropna()
        price_series_cub = comparables_df["precio_m2_cubierto"].dropna()
        detail_parts: List[str] = []

        if not price_series.empty:
            median_m2 = float(price_series.median())
            col_valor_m2.metric(
                "Valor estimado USD/m2",
                f"{median_m2:,.0f} USD/m2",
                delta=f"{comparables_len} comparables",
            )
            p25 = float(price_series.quantile(0.25))
            p75 = float(price_series.quantile(0.75))
            detail_parts.append(f"Total P25 {p25:,.0f} | P75 {p75:,.0f}")
        else:
            col_valor_m2.info("Sin valores por m2 en estas referencias.")

        if not price_series_cub.empty:
            median_m2_cub = float(price_series_cub.median())
            col_valor_m2_cub.metric(
                "Valor USD/m2 cubierto",
                f"{median_m2_cub:,.0f} USD/m2",
                delta=f"{comparables_len} comparables",
            )
            p25_c = float(price_series_cub.quantile(0.25))
            p75_c = float(price_series_cub.quantile(0.75))
            detail_parts.append(f"Cubierta P25 {p25_c:,.0f} | P75 {p75_c:,.0f}")
        else:
            col_valor_m2_cub.info("Sin datos de sup. cubierta en comparables.")

        median_distance = comparables_df["distance_km"].median()
        if pd.notna(median_distance):
            detail_parts.append(f"Radio mediano {float(median_distance):.2f} km")
        if detail_parts:
            col_valor_detalle.caption(" | ".join(detail_parts))
        else:
            col_valor_detalle.caption(
                "No hay estadisticas suficientes para resumir los comparables."
            )

    submitted = st.button("Estimar precio", type="primary")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "operacion": operacion,
                    "tipo_propiedad": tipo,
                    "loc_desc": localidad,
                    "superficie_total": superficie,
                    "superficie_cubierta": superficie_cubierta,
                    "banos": float(banos),
                    "dormitorios": float(dormitorios),
                    "cocheras_total": float(cocheras),
                    "prp_lat": float(latitud),
                    "prp_lng": float(longitud),
                    GEO_ZONE_COLUMN: barrio,
                }
            ]
        )
        prediction = artifacts.model.predict(input_df)[0]
        st.metric("Precio estimado (USD)", f"{prediction:,.0f}")
        notes = getattr(artifacts.model, "last_prediction_notes", [])
        for note in notes:
            st.warning(note)
        st.write(
            "Comparacion con la distribucion del dataset: "
            f"mediana {artifacts.price_summary['p50']:,.0f} USD."
        )
        if comparables_df.empty:
            st.info("No encontramos propiedades comparables para mostrar en la zona seleccionada.")
        else:
            st.subheader("Propiedades comparables cercanas")
            display_cols = [
                "operacion",
                "tipo_propiedad",
                "loc_desc",
                GEO_ZONE_COLUMN,
                "superficie_total",
                "superficie_cubierta",
                TARGET_COLUMN,
                "precio_m2",
                "precio_m2_cubierto",
                "distance_km",
                "dormitorios",
                "banos",
                "cocheras_total",
            ]
            available_cols = [col for col in display_cols if col in comparables_df.columns]
            table = (
                comparables_df[available_cols]
                .rename(
                    columns={
                        GEO_ZONE_COLUMN: "zona_geografica",
                        TARGET_COLUMN: "precio_usd",
                    }
                )
                .copy()
            )
            formatters = {
                "superficie_total": "{:,.0f}",
                "superficie_cubierta": "{:,.0f}",
                "precio_usd": "{:,.0f}",
                "precio_m2": "{:,.0f}",
                "precio_m2_cubierto": "{:,.0f}",
                "distance_km": "{:.2f}",
                "dormitorios": "{:,.0f}",
                "banos": "{:,.0f}",
                "cocheras_total": "{:,.0f}",
            }
            st.dataframe(
                table.style.format({k: v for k, v in formatters.items() if k in table.columns}),
                use_container_width=True,
            )

def main() -> None:
    st.set_page_config(
        page_title="Mercado inmobiliario Mendoza",
        page_icon=":house:",
        layout="wide",
    )
    st.title("Mercado inmobiliario de Mendoza")
    st.caption(
        "Explora los avisos procesados, navega las visualizaciones y proba el modelo "
        "de precio entrenado en las entregas previas."
    )

    artifacts = get_model_artifacts()
    operaciones, tipos = render_sidebar(artifacts)

    prepared = get_prepared_data()
    filtered_data = build_filtered_data(prepared, operaciones, tipos)

    tab_resumen, tab_viz, tab_modelo = st.tabs(
        ["Resumen", "Visualizaciones", "Modelo"]
    )
    with tab_resumen:
        render_summary(filtered_data, artifacts)
    with tab_viz:
        render_visualisations(filtered_data)
    with tab_modelo:
        render_prediction_ui(prepared, artifacts)


if __name__ == "__main__":
    main()

