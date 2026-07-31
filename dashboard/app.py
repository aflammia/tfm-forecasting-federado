"""
dashboard/app.py — panel de resultados del TFM (Fase 4, MLOps).

Lee los CSV que ya generan los scripts de src/ en reports/ (comparación de las 5 condiciones
A-E, baselines y el tuning de la Sesión 28) -- no reentrena ni recalcula nada. Como esos CSV
están gitignored (se generan localmente al correr el pipeline), este dashboard solo funciona
donde ya se ejecutaron esos scripts al menos una vez.

Ejecutar: streamlit run dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

REPORTS = Path(__file__).resolve().parents[1] / "reports"

st.set_page_config(page_title="TFM Federado -- Resultados", layout="wide")


@st.cache_data
def cargar_comparacion() -> pd.DataFrame:
    """Une todos los resultados_*.csv en una sola tabla método/split/métricas. Cada fichero ya
    fue producido y verificado por su propio script (ver RESEARCH_LOG) -- aquí solo se
    concatenan, sin recalcular nada."""
    ficheros = [
        ("resultados_baselines.csv", "baseline"),
        ("resultados_baselines_convencionales.csv", "baseline"),
        ("resultados_condicion_A_local.csv", "condicion"),
        ("resultados_condicion_B_silo.csv", "condicion"),
        ("resultados_condicion_C_global.csv", "condicion"),
        ("resultados_condicion_D_federado.csv", "condicion"),
        ("resultados_condicion_E_personalizacion.csv", "condicion"),
        ("resultados_condicion_E_original_sesion27.csv", "condicion"),
        ("resultados_correccion_duan.csv", "condicion"),
    ]
    columnas = ["metodo", "split", "wmape_media", "wmape_mediana", "mase_mediana",
                "rmsse_mediana", "cobertura"]
    partes = []
    for nombre_fichero, col_metodo in ficheros:
        ruta = REPORTS / nombre_fichero
        if not ruta.exists():
            continue
        df = pd.read_csv(ruta)
        df = df.rename(columns={col_metodo: "metodo"})
        partes.append(df[columnas])
    if not partes:
        return pd.DataFrame(columns=columnas)
    return pd.concat(partes, ignore_index=True).drop_duplicates(subset=["metodo", "split"])


@st.cache_data
def cargar_historial_rondas() -> pd.DataFrame:
    ruta = REPORTS / "historial_rondas_condicion_D.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


@st.cache_data
def cargar_por_serie() -> pd.DataFrame:
    ruta = REPORTS / "por_serie_condicion_E_test.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


st.title("TFM -- Previsión de demanda federada")
st.caption(
    "Datos leídos directamente de reports/*.csv, generados por los scripts de src/. "
    "Ningún número se recalcula aquí -- este panel solo visualiza resultados ya cerrados y "
    "documentados en docs/RESEARCH_LOG.md."
)

tab_comparacion, tab_convergencia, tab_series = st.tabs(
    ["Comparación de métodos", "Convergencia federada (D)", "Detalle por serie (E)"]
)

with tab_comparacion:
    comparacion = cargar_comparacion()
    if comparacion.empty:
        st.warning(
            "No se encontró ningún reports/resultados_*.csv. Ejecuta los scripts de src/ "
            "(10 a 19) para generarlos -- ver docs/STATE.md."
        )
    else:
        split = st.radio("Split", ["test", "val"], horizontal=True, index=0)
        vista = (
            comparacion[comparacion["split"] == split]
            .sort_values("wmape_mediana")
            .reset_index(drop=True)
        )
        col_tabla, col_grafico = st.columns([1, 1])
        with col_tabla:
            st.dataframe(
                vista[["metodo", "wmape_mediana", "mase_mediana", "rmsse_mediana", "cobertura"]]
                .style.format({
                    "wmape_mediana": "{:.4f}", "mase_mediana": "{:.4f}",
                    "rmsse_mediana": "{:.4f}", "cobertura": "{:.1%}",
                }),
                height=520,
            )
        with col_grafico:
            st.bar_chart(vista.set_index("metodo")["wmape_mediana"])
        st.caption(
            "Ordenado por WMAPE mediana (métrica principal del proyecto, robusta a series con "
            "ventas bajas/intermitentes -- T2.1, RESEARCH_LOG Sesión 18)."
        )

with tab_convergencia:
    historial = cargar_historial_rondas()
    if historial.empty:
        st.warning(
            "No se encontró reports/historial_rondas_condicion_D.csv. Ejecuta "
            "src/16_condicion_d_federado.py para generarlo."
        )
    else:
        configs = sorted(historial["config"].unique())
        elegidas = st.multiselect("Configuraciones", configs, default=configs)
        vista = historial[historial["config"].isin(elegidas)]
        pivote = vista.pivot(index="ronda", columns="config", values="val_loss")
        st.line_chart(pivote)
        st.caption(
            "Pérdida de validación centralizada (val global) por ronda de FedAvg/FedProx -- "
            "checkpointing manual, ver src/federado_flower.py."
        )

with tab_series:
    por_serie = cargar_por_serie()
    if por_serie.empty:
        st.warning(
            "No se encontró reports/por_serie_condicion_E_test.csv. Ejecuta "
            "src/17_condicion_e_personalizacion.py para generarlo."
        )
    else:
        tiendas = sorted(por_serie["store_nbr"].unique())
        tienda_elegida = st.selectbox("Tienda", ["(todas)"] + [str(t) for t in tiendas])
        vista = por_serie if tienda_elegida == "(todas)" else por_serie[
            por_serie["store_nbr"] == int(tienda_elegida)
        ]
        col_tabla, col_hist = st.columns([1, 1])
        with col_tabla:
            st.dataframe(
                vista.sort_values("wmape", ascending=False)
                .style.format({"wmape": "{:.3f}", "mase": "{:.3f}", "rmsse": "{:.3f}"}),
                height=460,
            )
        with col_hist:
            st.bar_chart(vista["wmape"].dropna())
        st.caption(
            "WMAPE/MASE/RMSSE por serie (tienda×familia) de la Condición E (test) -- "
            "sin predicciones semana a semana guardadas: solo las métricas agregadas por serie "
            "que produce src/17_condicion_e_personalizacion.py."
        )
