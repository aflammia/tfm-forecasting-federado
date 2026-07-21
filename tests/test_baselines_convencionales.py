"""
T2.2b — Tests de las piezas propias de src/11_baselines_convencionales.py.

No reajusta los 1.782 ETS ni los 54 LightGBM (correspondería a un test de integración, lento) --
verifica con datos sintéticos pequeños las dos piezas nuevas y propensas a error: la
reconstrucción/interpolación del calendario de entrada de ETS, y la degradación del modelo según
el historial disponible (Sesión 19: la razón de ser de este módulo es evitar que un hueco interno,
p.ej. Navidad, distorsione la estacionalidad de 52 semanas asumida por ETS).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from importlib import import_module

modulo = import_module("11_baselines_convencionales")


# ============================================================ recorte de prefijo de ceros (Sesión 21)

def test_recortar_ceros_iniciales_quita_el_prefijo():
    """Sesión 21: no todas las tiendas venden todas las familias desde el inicio de train (p.ej.
    surtido incorporado más tarde) -- ese prefijo de ceros estructurales debe recortarse, no
    tratarse como historial real de ventas bajas."""
    semanas = pd.date_range("2013-01-07", periods=6, freq="7D")
    data_serie = pd.DataFrame({
        "store_nbr": 1, "family": "PRODUCE", "week_start": semanas,
        "ventas": [0.0, 0.0, 0.0, 100.0, 120.0, 90.0],
    })
    recortada = modulo._recortar_ceros_iniciales(data_serie)
    assert len(recortada) == 3
    assert recortada["week_start"].min() == semanas[3]


def test_recortar_ceros_iniciales_serie_toda_cero_queda_vacia():
    semanas = pd.date_range("2013-01-07", periods=4, freq="7D")
    data_serie = pd.DataFrame({
        "store_nbr": 1, "family": "BOOKS", "week_start": semanas, "ventas": [0.0, 0.0, 0.0, 0.0],
    })
    recortada = modulo._recortar_ceros_iniciales(data_serie)
    assert recortada.empty


def test_recortar_ceros_iniciales_sin_prefijo_no_cambia_nada():
    semanas = pd.date_range("2013-01-07", periods=4, freq="7D")
    data_serie = pd.DataFrame({
        "store_nbr": 1, "family": "GROCERY I", "week_start": semanas,
        "ventas": [50.0, 60.0, 70.0, 80.0],
    })
    recortada = modulo._recortar_ceros_iniciales(data_serie)
    assert len(recortada) == 4


# ============================================================ reconstrucción del calendario de ETS

def test_serie_log_reindexada_interpola_el_hueco():
    """Una serie con una semana ausente en medio debe reaparecer, con log_ventas interpolado
    linealmente entre sus vecinas -- no debe quedar NaN ni desplazar las semanas siguientes."""
    semanas = pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-27", "2020-02-03"])  # falta 01-20
    data_serie = pd.DataFrame({
        "store_nbr": 1, "family": "A", "week_start": semanas,
        "ventas": [10.0, 20.0, 40.0, 50.0],
    })
    serie = modulo._serie_log_reindexada(data_serie)
    assert len(serie) == 5  # 4 originales + 1 semana reintroducida
    assert pd.Timestamp("2020-01-20") in serie.index
    assert not serie.isna().any()
    # interpolación lineal en escala log entre log1p(20) y log1p(40)
    esperado = (np.log1p(20.0) + np.log1p(40.0)) / 2
    assert np.isclose(serie.loc["2020-01-20"], esperado)


def test_serie_log_reindexada_sin_huecos_no_cambia_nada():
    semanas = pd.date_range("2020-01-06", periods=4, freq="7D")
    data_serie = pd.DataFrame({
        "store_nbr": 1, "family": "A", "week_start": semanas,
        "ventas": [10.0, 20.0, 30.0, 40.0],
    })
    serie = modulo._serie_log_reindexada(data_serie)
    assert len(serie) == 4
    assert np.allclose(serie.values, np.log1p([10.0, 20.0, 30.0, 40.0]))


# ============================================================ degradación de ETS según historial

def test_ets_devuelve_nan_si_no_hay_historial_suficiente():
    """Con menos de MIN_SEMANAS_MODELO observaciones no se puede ajustar ni un nivel simple --
    debe devolver NaN en vez de fallar o inventar un número."""
    serie = pd.Series([np.log1p(10.0)], index=pd.date_range("2020-01-06", periods=1, freq="7D"))
    pred = modulo._ajustar_y_pronosticar_ets(serie, pasos=16)
    assert len(pred) == 16
    assert np.isnan(pred).all()


def test_ets_con_historial_corto_no_estacional_devuelve_numeros_finitos():
    """Entre MIN_SEMANAS_MODELO y MIN_SEMANAS_TENDENCIA: debe degradar a nivel simple (sin
    tendencia ni estacionalidad) y aun así producir una predicción válida."""
    rng = np.random.RandomState(0)
    n = modulo.MIN_SEMANAS_MODELO + 1
    assert n < modulo.MIN_SEMANAS_TENDENCIA
    valores = np.log1p(50 + rng.normal(0, 2, n).cumsum().clip(min=-40))
    serie = pd.Series(valores, index=pd.date_range("2020-01-06", periods=n, freq="7D"))
    pred = modulo._ajustar_y_pronosticar_ets(serie, pasos=16)
    assert len(pred) == 16
    assert np.isfinite(pred).all()


def test_ets_horizonte_respetado():
    """El número de predicciones devueltas debe ser siempre exactamente `pasos`, cualquiera que
    sea el modelo elegido (nivel, tendencia o estacional)."""
    rng = np.random.RandomState(1)
    n = modulo.MIN_SEMANAS_ESTACIONAL + 5
    valores = np.log1p(50 + rng.normal(0, 2, n).cumsum().clip(min=-40))
    serie = pd.Series(valores, index=pd.date_range("2013-01-07", periods=n, freq="7D"))
    pred = modulo._ajustar_y_pronosticar_ets(serie, pasos=16)
    assert len(pred) == 16
    assert np.isfinite(pred).all()


# ============================================================ features de LightGBM

def test_features_lgbm_no_incluyen_columnas_normalizadas_ni_objetivo():
    """Los árboles no necesitan z-score (invariantes a transformaciones monótonas) -- y el
    objetivo (log_ventas) no debe colarse como feature de entrada."""
    assert "log_ventas" not in modulo.FEATURES_LGBM
    assert not any(c.endswith("_z") for c in modulo.FEATURES_LGBM)


def test_features_lgbm_existen_en_dataset_features():
    """Smoke test de esquema: si T1.4 renombrara/eliminara una columna, este test debe fallar
    antes que el pipeline de T2.2b en producción."""
    processed = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset_features.parquet"
    if not processed.exists():
        pytest.skip("dataset_features.parquet no generado en este entorno")
    columnas = pd.read_parquet(processed, columns=None).columns
    for c in modulo.FEATURES_LGBM:
        assert c in columnas, f"Columna esperada por LightGBM ausente en dataset_features.parquet: {c}"
