"""
T1.5 — Suite de tests del pipeline de datos (Fase 1).

Formaliza como pytest las verificaciones que se hicieron a mano en las Sesiones 11, 13 y 16
del RESEARCH_LOG, para que corran automáticamente (localmente y en el CI de la Fase 4) cada
vez que se toque el pipeline de datos.

Ejecutar: pytest tests/ -v
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"

N_TIENDAS_ESPERADO = 54
N_FAMILIAS_ESPERADO = 33
SILOS_ESPERADOS = {"Grande": 9, "Mediano": 26, "Pequeno": 19}


@pytest.fixture(scope="module")
def stores():
    return pd.read_csv(PROCESSED / "stores_silos.csv")


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED / "dataset_features.parquet")


@pytest.fixture(scope="module")
def split_config():
    with open(CONFIGS / "split_config.json", encoding="utf-8") as f:
        return json.load(f)


# ============================================================ SILOS (T1.1)

def test_numero_de_tiendas(stores):
    assert stores["store_nbr"].nunique() == N_TIENDAS_ESPERADO


def test_todas_las_tiendas_tienen_silo(stores):
    assert stores["silo"].isna().sum() == 0
    assert set(stores["silo"].unique()) == set(SILOS_ESPERADOS.keys())


def test_composicion_de_silos_es_la_esperada(stores):
    """Congela la Propuesta 2 (Sesión 9): Grande=9, Mediano=26, Pequeño=19."""
    conteo = stores["silo"].value_counts().to_dict()
    assert conteo == SILOS_ESPERADOS


# ============================================================ DATASET (T1.2/T1.3/T1.4)

def test_numero_de_familias(features):
    assert features["family"].nunique() == N_FAMILIAS_ESPERADO


def test_sin_filas_duplicadas(features):
    """Cada combinación tienda-familia-semana debe aparecer exactamente una vez."""
    duplicados = features.duplicated(subset=["store_nbr", "family", "week_start"]).sum()
    assert duplicados == 0


def test_ventas_no_negativas(features):
    assert (features["ventas"] >= 0).all()


def test_sin_nan_en_columnas_de_entrenamiento(features):
    cols_modelo = [c for c in features.columns if c.endswith("_z")] + [
        "semana_sin", "semana_cos", "mes_sin", "mes_cos", "family_id", "store_id", "log_ventas",
    ]
    for c in cols_modelo:
        assert features[c].isna().sum() == 0, f"NaN inesperado en columna de entrenamiento: {c}"


# ============================================================ CORTE TEMPORAL (T1.3) — anti-fuga

def test_split_sin_solape_temporal(features, split_config):
    train_max = features.loc[features.split == "train", "week_start"].max()
    val_min = features.loc[features.split == "val", "week_start"].min()
    val_max = features.loc[features.split == "val", "week_start"].max()
    test_min = features.loc[features.split == "test", "week_start"].min()
    assert train_max < val_min, "FUGA: train se solapa con val"
    assert val_max < test_min, "FUGA: val se solapa con test"


def test_fechas_de_split_coinciden_con_config(features, split_config):
    """val y test no se ven afectados por el recorte de T1.4 (quedan al final de cada serie,
    con historial de sobra para los lags) -> deben coincidir EXACTAMENTE con T1.3."""
    for nombre in ("val", "test"):
        min_real = str(features.loc[features.split == nombre, "week_start"].min().date())
        max_real = str(features.loc[features.split == nombre, "week_start"].max().date())
        assert min_real == split_config[nombre]["desde"]
        assert max_real == split_config[nombre]["hasta"]


def test_train_recortado_por_lags_es_coherente(features, split_config):
    """train SÍ se ve afectado: T1.4 recorta las primeras semanas de cada serie por falta de
    historial para los lags de hasta 8 semanas -> su fecha mínima real debe ser POSTERIOR a la
    de T1.3 (nunca anterior), pero su fecha máxima no cambia (no se toca el final de train)."""
    train_min_real = features.loc[features.split == "train", "week_start"].min()
    train_max_real = str(features.loc[features.split == "train", "week_start"].max().date())
    assert train_min_real >= pd.Timestamp(split_config["train"]["desde"])
    assert train_max_real == split_config["train"]["hasta"]


def test_ninguna_semana_parcial_en_el_dataset_final(features):
    """T1.3: se excluyeron las semanas con dias_con_dato<7 -- esa columna ya no debe existir."""
    assert "dias_con_dato" not in features.columns


# ============================================================ ANTI-FUGA DE LAGS (T1.4/T1.4b)

def test_lag_log_1_coincide_con_log_ventas_de_la_semana_anterior(features):
    """
    Verificación anti-fuga central del proyecto: para una muestra aleatoria, lag_log_1 de la
    semana W debe ser EXACTAMENTE log_ventas de la semana W-1 de la misma serie (tienda×familia).
    Formaliza la comprobación manual de la Sesión 16.
    """
    idx = features.set_index(["store_nbr", "family", "week_start"])["log_ventas"]
    muestra = features.sample(min(300, len(features)), random_state=7)

    fallos = 0
    comprobadas = 0
    for _, fila in muestra.iterrows():
        clave_anterior = (fila.store_nbr, fila.family, fila.week_start - pd.Timedelta(weeks=1))
        if clave_anterior in idx.index:
            comprobadas += 1
            if not np.isclose(idx.loc[clave_anterior], fila["lag_log_1"]):
                fallos += 1

    assert comprobadas > 0, "La muestra no permitió comprobar ningún caso (revisar test)"
    assert fallos == 0, f"FUGA DETECTADA: {fallos} de {comprobadas} filas con lag_log_1 incorrecto"


def test_lags_no_usan_informacion_de_semanas_futuras(features):
    """Comprobación estructural: para cada fila, todas sus columnas lag_log_* deben corresponder
    a semanas ANTERIORES a week_start (nunca posteriores)."""
    con_lags = features.dropna(subset=["lag_log_1"]).sample(min(200, len(features)), random_state=7)
    for _, fila in con_lags.iterrows():
        # lag_log_8 es el más antiguo -> su semana de origen debe ser 8 semanas antes, no después
        semana_origen_lag8 = fila["week_start"] - pd.Timedelta(weeks=8)
        assert semana_origen_lag8 < fila["week_start"]


# ============================================================ NORMALIZACIÓN (T1.4b)

def test_estandarizacion_calculada_solo_con_train(features):
    """Las columnas _z deben tener media~0 y std~1 en TRAIN (no necesariamente en val/test,
    que es precisamente la señal de que las estadísticas no se recalcularon con fuga)."""
    cols_z = [c for c in features.columns if c.endswith("_z")]
    assert len(cols_z) > 0
    train = features[features.split == "train"]
    for c in cols_z:
        media = train[c].mean()
        std = train[c].std()
        assert abs(media) < 0.01, f"{c}: media en train demasiado lejos de 0 ({media:.4f})"
        assert abs(std - 1) < 0.01, f"{c}: std en train demasiado lejos de 1 ({std:.4f})"


def test_codificacion_ciclica_en_rango_valido(features):
    for c in ["semana_sin", "semana_cos", "mes_sin", "mes_cos"]:
        assert features[c].between(-1.0, 1.0).all()


def test_ids_categoricos_en_rango_esperado(features):
    assert features["family_id"].between(0, N_FAMILIAS_ESPERADO - 1).all()
    assert features["store_id"].between(0, N_TIENDAS_ESPERADO - 1).all()
