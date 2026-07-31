"""
metrics.py — T2.1: módulo de métricas de evaluación, compartido por TODAS las condiciones (A-F).

Métricas implementadas:
  - WMAPE (Weighted MAPE) — métrica principal del proyecto.
  - RMSSE (Root Mean Squared Scaled Error) — la métrica oficial de la competición M5, escala
    el error por la dificultad "naive" de cada serie (permite comparar series de escalas
    muy distintas, relevante entre silos de tamaño tan dispar como los nuestros).
  - MASE (Mean Absolute Scaled Error) — el equivalente en error absoluto de RMSSE.

Además: cálculo por serie (tienda×familia) y agregado, y un test de Wilcoxon pareado para
comparar dos condiciones de forma estadísticamente rigurosa (Sesión 1, diseño experimental).

No es un script de pipeline — es una LIBRERÍA que importan los scripts de entrenamiento/evaluación:
    from metrics import wmape, mase, rmsse, metricas_por_serie, comparar_condiciones
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================ métricas base (arrays 1D)

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = suma de errores absolutos / suma de valores reales absolutos.
    Métrica principal del proyecto: no explota con ceros (a diferencia de MAPE clásico)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denominador = np.sum(np.abs(y_true))
    if denominador == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominador)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Mean Absolute Scaled Error. Escala el MAE del modelo por el MAE de un naive de un
    paso (venta de la semana anterior) calculado EN TRAIN de esa misma serie."""
    y_true, y_pred, y_train = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(y_train, float)
    escala = np.mean(np.abs(np.diff(y_train)))
    if escala == 0 or len(y_train) < 2:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / escala)


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Root Mean Squared Scaled Error -- la métrica oficial de la competición M5.
    Mismo principio que MASE, pero con error cuadrático en vez de absoluto."""
    y_true, y_pred, y_train = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(y_train, float)
    escala = np.mean(np.diff(y_train) ** 2)
    if escala == 0 or len(y_train) < 2:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / escala))


# ============================================================ cálculo por serie (tienda×familia)

def metricas_por_serie(
    df_eval: pd.DataFrame,
    df_train: pd.DataFrame,
    col_real: str = "ventas",
    col_pred: str = "prediccion",
    id_cols: tuple[str, ...] = ("store_nbr", "family"),
) -> pd.DataFrame:
    """
    Calcula WMAPE, MASE y RMSSE por cada serie (tienda×familia).

    df_eval: filas de val o test, con columnas id_cols + col_real + col_pred.
    df_train: filas de train de esas mismas series (para el denominador de escala de MASE/RMSSE).
    """
    filas = []
    for claves, grupo_eval in df_eval.groupby(list(id_cols)):
        mask_train = np.ones(len(df_train), dtype=bool)
        claves_iter = claves if isinstance(claves, tuple) else (claves,)
        for col, val in zip(id_cols, claves_iter):
            mask_train &= (df_train[col] == val)
        y_train_serie = df_train.loc[mask_train, col_real].values

        y_true = grupo_eval[col_real].values
        y_pred = grupo_eval[col_pred].values

        fila = dict(zip(id_cols, claves_iter))
        fila["n_semanas"] = len(y_true)
        fila["wmape"] = wmape(y_true, y_pred)
        fila["mase"] = mase(y_true, y_pred, y_train_serie)
        fila["rmsse"] = rmsse(y_true, y_pred, y_train_serie)
        filas.append(fila)

    return pd.DataFrame(filas)


def resumen(metricas_serie: pd.DataFrame, cols: tuple[str, ...] = ("wmape", "mase", "rmsse")) -> pd.Series:
    """Resumen agregado (media y mediana) de las métricas por serie -- para reportar un único
    número por condición experimental."""
    out = {}
    for c in cols:
        out[f"{c}_media"] = metricas_serie[c].mean()
        out[f"{c}_mediana"] = metricas_serie[c].median()
    return pd.Series(out)


# ============================================================ comparación estadística (Wilcoxon)

@dataclass
class ResultadoWilcoxon:
    metrica: str
    n_series: int
    mediana_a: float
    mediana_b: float
    estadistico: float
    p_valor: float
    significativo_05: bool

    def __repr__(self) -> str:
        signif = "SÍ" if self.significativo_05 else "no"
        return (f"Wilcoxon[{self.metrica}] n={self.n_series} "
                f"mediana_A={self.mediana_a:.4f} mediana_B={self.mediana_b:.4f} "
                f"p={self.p_valor:.4g} (significativo al 5%: {signif})")


def comparar_condiciones(
    metricas_a: pd.DataFrame,
    metricas_b: pd.DataFrame,
    metrica: str = "wmape",
    id_cols: tuple[str, ...] = ("store_nbr", "family"),
) -> ResultadoWilcoxon:
    """
    Test de Wilcoxon pareado (signed-rank) entre dos condiciones, sobre la métrica indicada,
    emparejando por serie (misma tienda×familia en ambas). Es la comprobación de significancia
    central del diseño experimental (Sesión 1 del RESEARCH_LOG): A vs D, C vs D, D vs E, etc.
    """
    merged = metricas_a.merge(metricas_b, on=list(id_cols), suffixes=("_a", "_b"))
    a = merged[f"{metrica}_a"].values
    b = merged[f"{metrica}_b"].values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]

    estadistico, p_valor = stats.wilcoxon(a, b)
    return ResultadoWilcoxon(
        metrica=metrica,
        n_series=len(a),
        mediana_a=float(np.median(a)),
        mediana_b=float(np.median(b)),
        estadistico=float(estadistico),
        p_valor=float(p_valor),
        significativo_05=bool(p_valor < 0.05),
    )


def porcentaje_brecha_recuperada(wmape_local: float, wmape_centralizado: float, wmape_federado: float) -> float:
    """La 'métrica estrella' del proyecto (PLAN.md, sección 0):
    % de la brecha Local -> Centralizado que recupera el Federado."""
    brecha_total = wmape_local - wmape_centralizado
    if brecha_total == 0:
        return np.nan
    brecha_recuperada = wmape_local - wmape_federado
    return float(brecha_recuperada / brecha_total * 100)
