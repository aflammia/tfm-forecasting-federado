"""
T2.1 — Tests del módulo de métricas (src/metrics.py), con casos calculados a mano.

Ningún resultado se da por bueno sin verificarlo contra un cálculo independiente.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import (
    comparar_condiciones,
    mase,
    metricas_por_serie,
    porcentaje_brecha_recuperada,
    rmsse,
    wmape,
)

# ============================================================ WMAPE

def test_wmape_caso_calculado_a_mano():
    # |110-100| + |90-100| = 20 ; |100|+|100| = 200 -> 20/200 = 0.10
    y_true = np.array([100, 100])
    y_pred = np.array([110, 90])
    assert np.isclose(wmape(y_true, y_pred), 0.10)


def test_wmape_prediccion_perfecta_es_cero():
    y = np.array([5, 10, 15])
    assert np.isclose(wmape(y, y), 0.0)


def test_wmape_todo_cero_da_nan():
    y = np.array([0, 0])
    assert np.isnan(wmape(y, y + 1))  # sum(|y_true|)=0 -> indefinido, no debe devolver un numero falso


# ============================================================ MASE

def test_mase_caso_calculado_a_mano():
    # train=[10,12,14,16] -> diffs=[2,2,2] -> escala=2
    # eval: y_true=[20,22], y_pred=[21,21] -> MAE=(1+1)/2=1 -> MASE=1/2=0.5
    y_train = np.array([10, 12, 14, 16])
    y_true = np.array([20, 22])
    y_pred = np.array([21, 21])
    assert np.isclose(mase(y_true, y_pred, y_train), 0.5)


def test_mase_peor_que_naive_da_mayor_que_uno():
    y_train = np.array([10, 12, 14, 16])  # escala = 2
    y_true = np.array([20, 22])
    y_pred = np.array([25, 27])  # error de 5 en ambos -> MAE=5 -> MASE=2.5
    assert mase(y_true, y_pred, y_train) > 1.0


# ============================================================ RMSSE

def test_rmsse_caso_calculado_a_mano():
    # train=[10,12,14,16] -> diffs^2=[4,4,4] -> escala=4
    # eval: y_true=[20,22], y_pred=[21,21] -> RMSE=sqrt((1+1)/2)=1 -> RMSSE=1/sqrt(4)=0.5
    y_train = np.array([10, 12, 14, 16])
    y_true = np.array([20, 22])
    y_pred = np.array([21, 21])
    assert np.isclose(rmsse(y_true, y_pred, y_train), 0.5)


# ============================================================ métricas por serie (agregación)

def test_metricas_por_serie_dos_series_conocidas():
    df_train = pd.DataFrame({
        "store_nbr": [1, 1, 1, 1, 2, 2, 2, 2],
        "family": ["A"] * 4 + ["A"] * 4,
        "ventas": [10, 12, 14, 16, 100, 120, 140, 160],
    })
    df_eval = pd.DataFrame({
        "store_nbr": [1, 1, 2, 2],
        "family": ["A", "A", "A", "A"],
        "ventas": [20, 22, 200, 220],
        "prediccion": [21, 21, 210, 210],
    })
    resultado = metricas_por_serie(df_eval, df_train)
    assert len(resultado) == 2  # dos series: (1,A) y (2,A)
    fila_1 = resultado[resultado.store_nbr == 1].iloc[0]
    assert np.isclose(fila_1["mase"], 0.5)  # mismo caso que test_mase_caso_calculado_a_mano
    assert np.isclose(fila_1["rmsse"], 0.5)


# ============================================================ comparación estadística (Wilcoxon)

def test_wilcoxon_detecta_diferencia_real():
    """Si B es sistemáticamente mejor (wmape más bajo) en todas las series, debe salir significativo."""
    n = 30
    metricas_a = pd.DataFrame({
        "store_nbr": range(n), "family": ["X"] * n,
        "wmape": np.random.RandomState(0).uniform(0.20, 0.30, n),
    })
    metricas_b = pd.DataFrame({
        "store_nbr": range(n), "family": ["X"] * n,
        "wmape": metricas_a["wmape"] - 0.05,  # sistematicamente mejor
    })
    resultado = comparar_condiciones(metricas_a, metricas_b, metrica="wmape")
    assert resultado.significativo_05
    assert resultado.mediana_b < resultado.mediana_a


def test_wilcoxon_no_detecta_diferencia_si_es_ruido_simetrico():
    """Si la diferencia entre A y B es ruido aleatorio SIN sesgo sistemático, no debe ser significativo."""
    n = 30
    rng = np.random.RandomState(1)
    base = rng.uniform(0.20, 0.30, n)
    metricas_a = pd.DataFrame({"store_nbr": range(n), "family": ["X"] * n, "wmape": base})
    metricas_b = pd.DataFrame({
        "store_nbr": range(n), "family": ["X"] * n,
        "wmape": base + rng.normal(0, 0.001, n),  # ruido simetrico muy pequeño, sin sesgo
    })
    resultado = comparar_condiciones(metricas_a, metricas_b, metrica="wmape")
    assert not resultado.significativo_05


# ============================================================ métrica estrella del proyecto

def test_porcentaje_brecha_recuperada_caso_calculado_a_mano():
    # local=0.30, centralizado=0.10 (techo) -> brecha total=0.20
    # federado=0.12 -> recupera 0.18 de 0.20 -> 90%
    resultado = porcentaje_brecha_recuperada(wmape_local=0.30, wmape_centralizado=0.10, wmape_federado=0.12)
    assert np.isclose(resultado, 90.0)


def test_porcentaje_brecha_recuperada_federado_igual_a_local_es_cero():
    resultado = porcentaje_brecha_recuperada(wmape_local=0.30, wmape_centralizado=0.10, wmape_federado=0.30)
    assert np.isclose(resultado, 0.0)


def test_porcentaje_brecha_recuperada_federado_iguala_centralizado_es_cien():
    resultado = porcentaje_brecha_recuperada(wmape_local=0.30, wmape_centralizado=0.10, wmape_federado=0.10)
    assert np.isclose(resultado, 100.0)
