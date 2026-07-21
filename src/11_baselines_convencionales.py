"""
11_baselines_convencionales.py — T2.2b: baselines convencionales de retail (responde RQ2).

Los baselines "ingenuos" de T2.2 (persistencia, estacional, media móvil) no son un estándar de
la industria -- solo un suelo de cordura. Lo que responde RQ2 ("¿supera el federado a los
métodos convencionales de forecasting en retail?") son dos métodos con evidencia reciente
(Petropoulos, Grushka-Cockayne, Siemsen & Spiliotis, 2024 -- ver docs/REFERENCIAS.md):

  - ETS / Holt-Winters (suavizado exponencial con tendencia y estacionalidad) -- un modelo
    estadístico clásico, por serie (tienda×familia).
  - LightGBM -- gradient boosting sobre árboles, el estándar de facto en competiciones de
    forecasting como M5.

CORRECCIÓN (Sesión 22): el diseño original entrenaba un LightGBM POR TIENDA (54 modelos). Sin
ajuste de hiperparámetros ni early stopping, ese diseño no llegaba a superar ni siquiera al
baseline ingenuo más simple (media móvil de 4 semanas, T2.2) -- WMAPE test 0,394 vs 0,241. Al
tunear hiperparámetros correctamente se detectó la causa: cada modelo por tienda entrena con muy
pocos datos (~5-6 mil filas) y su early stopping decide sobre un val todavía más pequeño
(~264 filas), demasiado ruidoso para una decisión estable. Un ÚNICO modelo LightGBM GLOBAL
(`store_id` y `family_id` como categóricas, en vez de un modelo por tienda) tiene ~54x más datos
de entrenamiento y bate con claridad al baseline en las tres métricas por mediana. Es, además, el
mismo principio que motiva todo este TFM: compartir señal entre entidades (aquí, tiendas; en
Fase 3, silos vía FedAvg) generaliza mejor que entrenar cada una por separado con poco dato.

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
import json
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
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

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
    "es_festivo_nacional", "es_festivo_regional", "es_festivo_local", "semana_con_dia_pago",
    "family_id",
]

# Sesión 22: espacio de búsqueda de hiperparámetros para el tuning de LightGBM. El primer
# LightGBM (Sesión 21) usaba una configuración fija razonable pero SIN ajustar -- resultó no
# batir ni siquiera a la media móvil de 4 semanas (T2.2) en WMAPE. Este espacio cubre los
# hiperparámetros con más impacto en un GBM sobre un dataset tabular de tamaño modesto.
ESPACIO_BUSQUEDA_LGBM = {
    "objective": ["regression", "regression_l1"],
    "learning_rate": [0.01, 0.02, 0.05, 0.08],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [5, 15, 30],
    "feature_fraction": [0.7, 0.85, 1.0],
    "bagging_fraction": [0.7, 0.85, 1.0],
    "reg_lambda": [0.0, 0.5, 2.0],
}


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


# ============================================================ LightGBM (tuneado y global, Sesión 22)

def _wmape_natural(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """WMAPE agregado en escala natural de ventas, a partir de predicciones en log_ventas --
    la métrica que de verdad importa (no la pérdida de entrenamiento en escala log)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan


def buscar_hiperparametros_lgbm(features: pd.DataFrame, n_candidatos: int = 25, semilla: int = 42) -> dict:
    """Búsqueda aleatoria de hiperparámetros sobre el dataset AGRUPADO de las 54 tiendas (con
    `store_id` como feature categórica) -- el mismo dataset agrupado que usa el modelo FINAL
    (ver `predicciones_lightgbm`), así que esta búsqueda ya optimiza para la configuración que
    realmente se despliega, no para un proxy distinto.

    Selección por WMAPE en escala natural sobre val (no por la pérdida de entrenamiento en log) --
    es la métrica que se usa para comparar contra los demás baselines."""
    train = features[features.split == "train"]
    val = features[features.split == "val"]
    cols = FEATURES_LGBM + ["store_id"]

    claves = list(ESPACIO_BUSQUEDA_LGBM.keys())
    candidatos = []
    for i in range(n_candidatos):
        rng_i = np.random.RandomState(semilla + i)  # semilla independiente por candidato
        candidatos.append({k: rng_i.choice(ESPACIO_BUSQUEDA_LGBM[k]) for k in claves})

    mejor_config, mejor_wmape = None, np.inf
    print(f"Buscando hiperparámetros de LightGBM ({n_candidatos} candidatos, dataset agrupado)...")
    for i, config in enumerate(candidatos, start=1):
        config = {k: (v.item() if hasattr(v, "item") else v) for k, v in config.items()}
        modelo = lgb.LGBMRegressor(n_estimators=500, verbosity=-1, random_state=semilla, **config)
        modelo.fit(
            train[cols], train["log_ventas"],
            categorical_feature=["family_id", "store_id"],
            eval_set=[(val[cols], val["log_ventas"])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        wmape = _wmape_natural(val["log_ventas"].values, modelo.predict(val[cols]))
        if wmape < mejor_wmape:
            mejor_wmape, mejor_config = wmape, config
        if i % 5 == 0:
            print(f"  {i}/{n_candidatos} candidatos -- mejor WMAPE(val) hasta ahora: {mejor_wmape:.4f}")

    print(f"Mejor configuración (WMAPE val agrupado={mejor_wmape:.4f}): {mejor_config}")
    return mejor_config


def predicciones_lightgbm(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """UN ÚNICO modelo LightGBM GLOBAL (no por tienda), con `store_id` y `family_id` como
    categóricas, entrenado con TODAS las filas de train y prediciendo log_ventas. No hace falta
    normalizar (los árboles son invariantes a transformaciones monótonas) -- se usan las
    columnas log_* sin z-score.

    Corrección de diseño (Sesión 22): la primera versión (Sesión 21) entrenaba un modelo POR
    TIENDA (54 modelos) -- con hiperparámetros sin ajustar, ni siquiera batía al baseline
    ingenuo más simple (media móvil, T2.2). Al tunear correctamente se detectó la causa: cada
    modelo por tienda entrena con muy pocos datos (~5-6 mil filas) y decide cuándo parar
    (early stopping) sobre un val todavía más pequeño (~264 filas) -- demasiado ruidoso para una
    decisión estable. Un modelo GLOBAL tiene ~54x más datos de entrenamiento y bate con claridad
    al baseline en las tres métricas por mediana (ver RESEARCH_LOG Sesión 22). El número de
    árboles se decide por early stopping sobre TODO val (mucho más estable que el val de una
    sola tienda).

    Nota metodológica: al usar val para decidir cuándo parar, val deja de ser una estimación
    limpia de generalización para ESTE modelo (sí lo sigue siendo para
    persistencia/estacional/media móvil/ETS, que no usan val en su ajuste) -- la comparación
    justa contra los demás baselines es la de TEST, que nunca se toca durante el ajuste ni la
    búsqueda de hiperparámetros."""
    cols = FEATURES_LGBM + ["store_id"]
    train = features[features.split == "train"]
    val = features[features.split == "val"]
    eval_ = features[features.split.isin(["val", "test"])]

    print("Entrenando LightGBM global (tuneado, store_id + family_id como categóricas)...")
    modelo = lgb.LGBMRegressor(n_estimators=5000, verbosity=-1, random_state=42, **config)
    modelo.fit(
        train[cols], train["log_ventas"],
        categorical_feature=["family_id", "store_id"],
        eval_set=[(val[cols], val["log_ventas"])],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  árboles usados (early stopping): {modelo.best_iteration_}")

    pred_log = modelo.predict(eval_[cols])
    predicciones = eval_[["store_nbr", "family", "week_start", "split"]].copy()
    predicciones["pred_lightgbm"] = np.clip(np.expm1(pred_log), 0, None)  # ventas no negativas
    return predicciones


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

    config_lgbm = buscar_hiperparametros_lgbm(features)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    with open(CONFIGS / "lightgbm_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(config_lgbm, f, indent=2, ensure_ascii=False)
    pred_lgbm = predicciones_lightgbm(features, config_lgbm)

    filas_resumen = []
    for nombre, preds, col, fuente in [
        ("ETS/Holt-Winters", pred_ets, "pred_ets", modelado),
        ("LightGBM (global)", pred_lgbm, "pred_lightgbm", features),
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
