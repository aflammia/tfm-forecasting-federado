"""
11_baselines_convencionales.py — T2.2b: baselines convencionales de retail (responde RQ2).

Los baselines "ingenuos" de T2.2 (persistencia, estacional, media móvil) no son un estándar de
la industria -- solo un suelo de cordura. Lo que responde RQ2 ("¿supera el federado a los
métodos convencionales de forecasting en retail?") son dos métodos con evidencia reciente
(Petropoulos, Grushka-Cockayne, Siemsen & Spiliotis, 2024 -- ver docs/REFERENCIAS.md):

  - ETS / Holt-Winters (suavizado exponencial con tendencia y estacionalidad) -- un modelo
    estadístico clásico, por serie (tienda×familia).
  - LightGBM por tienda -- gradient boosting sobre árboles, el estándar de facto en
    competiciones de forecasting como M5. Un modelo por tienda (no global, no por serie
    individual), entrenado con las filas de todas sus familias -- así aprende patrones
    compartidos entre familias de la misma tienda, igual que hará luego el modelo local (A).

Ambos se entrenan SOLO con train y se evalúan sobre val/test con el módulo de métricas de T2.1,
en escala natural de ventas (expm1 de la predicción en log) -- comparable directamente con
reports/resultados_baselines.csv (T2.2) y, más adelante, con las condiciones A-F.

ETS usa dataset_modelado.parquet (T1.3), no dataset_features.parquet (T1.4): ETS no necesita
lags ya calculados, solo la serie temporal real, y dataset_modelado.parquet conserva el
historial completo de train (T1.4 recorta las primeras ~8 semanas de cada serie por el
warm-up de sus lags -- una pérdida innecesaria para un modelo que no usa esos lags).

Igual que en T2.2/T1.4 (Sesión 19): antes de ajustar ETS, cada serie se reindexa a un
calendario semanal completo (calendario_semanal.py) porque el 25-dic no tiene ninguna fila en
train.csv (tiendas cerradas) y eso deja huecos internos en casi todas las series -- un ETS
ajustado directamente sobre las filas tal cual (sin reindexar) asumiría implícitamente que las
semanas son consecutivas, distorsionando la estacionalidad de 52 semanas. Los huecos (pocos por
serie) se interpolan linealmente solo para dar continuidad a la serie de entrada del modelo.
"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import metricas_por_serie, resumen
from calendario_semanal import construir_calendario_completo

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"

N_VAL = 8
N_TEST = 8
HORIZONTE = N_VAL + N_TEST

MIN_SEMANAS_ESTACIONAL = 104  # >=2 ciclos anuales completos para estimar estacionalidad de 52 sem.
MIN_SEMANAS_TENDENCIA = 10
MIN_SEMANAS_MODELO = 2

FEATURES_LGBM = [
    "lag_log_1", "lag_log_2", "lag_log_4", "lag_log_8",
    "media_movil_log_4", "media_movil_log_8", "std_log_4",
    "log_onpromotion", "oil_price",
    "semana_sin", "semana_cos", "mes_sin", "mes_cos",
    "family_id",
]


# ============================================================ ETS por serie

def _recortar_ceros_iniciales(data_serie: pd.DataFrame) -> pd.DataFrame:
    """Recorta el prefijo de semanas con ventas=0 al inicio de la serie -- análogo a la fecha
    de apertura de T1.2 (05_build_modeling_dataset.py), pero a nivel tienda×familia en vez de
    solo tienda: no todas las tiendas venden todas las familias (p.ej. una tienda pequeña
    puede no tener surtido de BOOKS o LADIESWEAR, o una familia perecedera como PRODUCE puede
    incorporarse al surtido bien entrado el periodo de train). Ese prefijo de ceros no es
    ruido, es ausencia estructural de venta, y distorsiona gravemente el nivel/tendencia
    inicial que ETS estima (Sesión 21 del RESEARCH_LOG: 779 de 1.749 series -45%- tienen un
    prefijo de más de 4 semanas; sin recortarlo, ETS producía predicciones de millones de
    unidades en series como PRODUCE)."""
    con_venta = data_serie[data_serie["ventas"] > 0]
    if con_venta.empty:
        return data_serie.iloc[0:0]
    return data_serie[data_serie["week_start"] >= con_venta["week_start"].min()]


def _serie_log_reindexada(data_serie: pd.DataFrame) -> pd.Series:
    """Reconstruye el calendario semanal completo de UNA serie, ya recortada de su prefijo de
    ceros (huecos internos = NaN, interpolados linealmente para dar continuidad a la entrada
    de ETS)."""
    completo = construir_calendario_completo(data_serie, cols_valor=["ventas"])
    log_ventas = np.log1p(completo["ventas"])
    log_ventas.index = pd.DatetimeIndex(completo["week_start"])
    return log_ventas.interpolate(method="linear")


def _ajustar_y_pronosticar_ets(serie_log: pd.Series, pasos: int) -> np.ndarray:
    """Ajusta ETS aditivo sobre log_ventas, degradando el modelo si no hay historial
    suficiente para el componente que le corresponde: estacional+tendencia -> solo
    tendencia -> nivel simple. Devuelve `pasos` predicciones en escala log.

    `damped_trend=True` en ambas ramas con tendencia: se detectó empíricamente (Sesión 21 del
    RESEARCH_LOG) que una tendencia NO amortiguada se extrapola linealmente sin límite, y al
    deshacer el log con expm1 ese error lineal se vuelve EXPONENCIAL en escala de ventas -- con
    horizonte de 16 semanas, unas pocas series (p.ej. PRODUCE en varias tiendas) producían
    predicciones de millones de unidades frente a ventas reales de miles. Amortiguar la
    tendencia es la corrección estándar para este modo de fallo conocido de Holt-Winters en
    horizontes largos."""
    n = len(serie_log)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n >= MIN_SEMANAS_ESTACIONAL:
            modelo = ExponentialSmoothing(
                serie_log, trend="add", damped_trend=True, seasonal="add", seasonal_periods=52,
                initialization_method="estimated",
            ).fit()
        elif n >= MIN_SEMANAS_TENDENCIA:
            modelo = ExponentialSmoothing(
                serie_log, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            ).fit()
        elif n >= MIN_SEMANAS_MODELO:
            modelo = ExponentialSmoothing(
                serie_log, trend=None, seasonal=None, initialization_method="estimated",
            ).fit()
        else:
            return np.full(pasos, np.nan)
        return modelo.forecast(pasos).values


def predicciones_ets(modelado: pd.DataFrame) -> pd.DataFrame:
    """Una predicción por serie (tienda×familia) para cada semana de val+test, con un único
    ajuste sobre el historial de train (forecast multi-step, no reentrenamiento semanal --
    el estándar en la evaluación de baselines de forecasting de horizonte fijo)."""
    filas = []
    train = modelado[modelado.split == "train"]
    semanas_val_test = (
        modelado.loc[modelado.split.isin(["val", "test"]), ["week_start", "split"]]
        .drop_duplicates().sort_values("week_start").reset_index(drop=True)
    )
    assert len(semanas_val_test) == HORIZONTE, (
        f"Se esperaban {HORIZONTE} semanas de val+test, hay {len(semanas_val_test)}"
    )

    series = list(train.groupby(["store_nbr", "family"], sort=False))
    print(f"Ajustando ETS para {len(series)} series...")
    for i, ((store, family), g) in enumerate(series, start=1):
        g = _recortar_ceros_iniciales(g)
        if g.empty:
            fila = semanas_val_test.copy()
            fila["store_nbr"] = store
            fila["family"] = family
            fila["pred_ets"] = np.nan
            filas.append(fila)
            continue
        serie_log = _serie_log_reindexada(g)
        pred_log = _ajustar_y_pronosticar_ets(serie_log, HORIZONTE)
        # techo de cordura: incluso con tendencia amortiguada, la estacionalidad estimada sobre
        # un historial corto y post-lanzamiento irregular (p.ej. PRODUCE, Sesión 21) puede
        # quedar mal ajustada y producir predicciones varios órdenes de magnitud por encima de
        # cualquier valor observado. Un forecast a 300x el máximo histórico no es un punto
        # razonable bajo ninguna interpretación -- se acota a un múltiplo generoso (3x) del
        # máximo histórico de la propia serie, práctica estándar de guardrail en forecasting
        # productivo. No aplica a LightGBM (los árboles no pueden extrapolar fuera del rango de
        # los datos de entrenamiento, por construcción).
        techo = g["ventas"].max() * 3
        pred_ventas = np.clip(np.expm1(pred_log), 0, techo)  # ventas no pueden ser negativas
        fila = semanas_val_test.copy()
        fila["store_nbr"] = store
        fila["family"] = family
        fila["pred_ets"] = pred_ventas
        filas.append(fila)
        if i % 300 == 0:
            print(f"  {i}/{len(series)} series procesadas")

    return pd.concat(filas, ignore_index=True)


# ============================================================ LightGBM por tienda

def predicciones_lightgbm(features: pd.DataFrame) -> pd.DataFrame:
    """Un modelo LightGBM por tienda (54 modelos), entrenado con las filas de TRAIN de todas
    sus familias, prediciendo log_ventas. No hace falta normalizar (los árboles son invariantes
    a transformaciones monótonas de las features) -- se usan las columnas log_* sin z-score."""
    filas = []
    tiendas = sorted(features["store_nbr"].unique())
    print(f"Entrenando LightGBM para {len(tiendas)} tiendas...")
    for store in tiendas:
        d = features[features.store_nbr == store]
        train = d[d.split == "train"]
        eval_ = d[d.split.isin(["val", "test"])]
        if len(train) < 20 or len(eval_) == 0:
            continue

        modelo = lgb.LGBMRegressor(
            objective="regression", n_estimators=200, learning_rate=0.05,
            num_leaves=15, min_child_samples=5, verbosity=-1, random_state=42,
        )
        modelo.fit(
            train[FEATURES_LGBM], train["log_ventas"],
            categorical_feature=["family_id"],
        )
        pred_log = modelo.predict(eval_[FEATURES_LGBM])

        fila = eval_[["store_nbr", "family", "week_start", "split"]].copy()
        fila["pred_lightgbm"] = np.clip(np.expm1(pred_log), 0, None)  # ventas no pueden ser negativas
        filas.append(fila)

    return pd.concat(filas, ignore_index=True)


# ============================================================ evaluación (T2.1)

def evaluar(data_completa: pd.DataFrame, predicciones: pd.DataFrame, col_pred: str, split: str) -> pd.Series:
    df_train = data_completa[data_completa.split == "train"][["store_nbr", "family", "ventas"]]
    pred_split = predicciones[predicciones.split == split]
    df_eval = pred_split.merge(
        data_completa[["store_nbr", "family", "week_start", "ventas"]],
        on=["store_nbr", "family", "week_start"], how="left",
    )
    df_eval = df_eval.dropna(subset=[col_pred]).rename(columns={col_pred: "prediccion"})

    por_serie = metricas_por_serie(df_eval, df_train)
    n_esperado = data_completa[data_completa.split == split][["store_nbr", "family"]].drop_duplicates().shape[0]
    cobertura = len(por_serie) / n_esperado
    resumen_metricas = resumen(por_serie)
    resumen_metricas["cobertura"] = cobertura
    return resumen_metricas


def main() -> None:
    modelado = pd.read_parquet(PROCESSED / "dataset_modelado.parquet")
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")

    pred_ets = predicciones_ets(modelado)
    pred_lgbm = predicciones_lightgbm(features)

    filas_resumen = []
    for nombre, preds, col, fuente in [
        ("ETS/Holt-Winters", pred_ets, "pred_ets", modelado),
        ("LightGBM (por tienda)", pred_lgbm, "pred_lightgbm", features),
    ]:
        print(f"\n{'=' * 60}\n{nombre}\n{'=' * 60}")
        for split in ["val", "test"]:
            resumen_split = evaluar(fuente, preds, col, split)
            print(f"  [{split}] cobertura={resumen_split['cobertura']*100:.1f}%  "
                  f"WMAPE_media={resumen_split['wmape_media']:.4f}  "
                  f"MASE_media={resumen_split['mase_media']:.4f}  "
                  f"RMSSE_media={resumen_split['rmsse_media']:.4f}")
            fila = {"baseline": nombre, "split": split}
            fila.update(resumen_split.to_dict())
            filas_resumen.append(fila)

    df_resumen = pd.DataFrame(filas_resumen)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_baselines_convencionales.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nResumen guardado en: {out}")
    print("\n" + df_resumen.to_string(index=False))


if __name__ == "__main__":
    main()
