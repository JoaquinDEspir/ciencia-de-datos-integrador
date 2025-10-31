"""Streamlit app to explore Mendoza real-estate data and test the price model."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st

from app.model import ModelArtifacts, TARGET_COLUMN, train_price_model
from visualizations.altair_charts import (
    PreparedData,
    load_properties_data,
    price_density_map,
    price_distribution_chart,
    price_vs_surface_chart,
)


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
    return PreparedData(full=df, with_price_usd=with_price, with_geo=with_geo)


def render_visualisations(filtered: PreparedData) -> None:
    try:
        st.altair_chart(price_distribution_chart(filtered), use_container_width=True)
        st.altair_chart(price_vs_surface_chart(filtered), use_container_width=True)
        st.altair_chart(price_density_map(filtered), use_container_width=True)
    except (RuntimeError, ValueError) as exc:
        st.warning(str(exc))


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


def render_prediction_ui(artifacts: ModelArtifacts) -> None:
    st.subheader("Ingresar una propiedad para estimar precio in situ")
    with st.form("prediction_form"):
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
        superficie = st.number_input(
            "Superficie total (m2)", min_value=10.0, max_value=5000.0, value=120.0, step=10.0
        )
        dormitorios = st.slider("Dormitorios", min_value=0, max_value=10, value=2)
        banos = st.slider("Banos", min_value=1, max_value=8, value=2)
        cocheras = st.slider("Cocheras", min_value=0, max_value=5, value=1)
        fotos = st.slider("Cantidad de fotos", min_value=0, max_value=40, value=10)
        submitted = st.form_submit_button("Estimar precio")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "operacion": operacion,
                    "tipo_propiedad": tipo,
                    "loc_desc": localidad,
                    "superficie_total": superficie,
                    "banos": float(banos),
                    "dormitorios": float(dormitorios),
                    "cocheras_total": float(cocheras),
                    "fotos_cantidad": float(fotos),
                }
            ]
        )
        prediction = artifacts.model.predict(input_df)[0]
        st.metric("Precio estimado (USD)", f"{prediction:,.0f}")
        st.write(
            "Comparacion con la distribucion del dataset: "
            f"mediana {artifacts.price_summary['p50']:,.0f} USD."
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
        render_prediction_ui(artifacts)


if __name__ == "__main__":
    main()

