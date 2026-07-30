"""
19_correccion_duan.py — Etapa 4 del plan de tuning (Sesión 28): corrección de Duan sobre la
mejor variante de personalización encontrada en la Etapa 3.

Reentrena la MISMA variante ganadora de `resultados_condicion_E_personalizacion.csv` (no hace
falta guardar los 54 modelos de la Etapa 3 a disco -- entrenar de nuevo esos 54 modelos es barato
comparado con las simulaciones federadas), esta vez guardando también las predicciones en escala
log sobre TRAIN para calcular el factor de Duan (`correccion_sesgo.py`) por tienda, y comparando
WMAPE con y sin la corrección.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import entrenar, predecir_log
from correccion_sesgo import aplicar_correccion_duan, factor_correccion_duan
from metrics import metricas_por_serie, resumen

import importlib
m17 = importlib.import_module("17_condicion_e_personalizacion")

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def personalizar_con_y_sin_duan(features: pd.DataFrame, modelo_global, lr: float, congelar_base: bool) -> pd.DataFrame:
    filas = []
    tiendas = sorted(features["store_nbr"].unique())
    print(f"Personalizando con seguimiento de corrección de Duan para {len(tiendas)} tiendas...")
    for i, store in enumerate(tiendas, start=1):
        d = features[features.store_nbr == store]
        train = d[d.split == "train"]
        val = d[d.split == "val"]
        eval_ = d[d.split.isin(["val", "test"])]
        if len(train) < 20 or len(eval_) == 0:
            continue
        if len(val) < 10:
            corte = int(len(train) * 0.85)
            train, val = train.iloc[:corte], train.iloc[corte:]
            if len(val) < 5:
                val = train

        modelo, hist = entrenar(train, val, epochs=50, paciencia=10, lr=lr,
                                 modelo_inicial=modelo_global, congelar_base=congelar_base)

        pred_log_train = predecir_log(modelo, train)
        factor = factor_correccion_duan(train["log_ventas"].values, pred_log_train)

        pred_log_eval = predecir_log(modelo, eval_)
        fila = eval_[["store_nbr", "family", "week_start", "split"]].copy()
        fila["pred_sin_corregir"] = np.clip(np.expm1(pred_log_eval), 0, None)
        fila["pred_corregida"] = aplicar_correccion_duan(pred_log_eval, factor)
        fila["factor_duan"] = factor
        filas.append(fila)

        if i % 15 == 0:
            print(f"  {i}/{len(tiendas)} tiendas (factor de Duan más reciente: {factor:.4f})")

    return pd.concat(filas, ignore_index=True)


def evaluar(predicciones: pd.DataFrame, features: pd.DataFrame, col_pred: str, nombre: str) -> pd.DataFrame:
    df_train_ref = features[features.split == "train"][["store_nbr", "family", "ventas"]]
    filas_resumen = []
    for split in ["val", "test"]:
        pred_split = predicciones[predicciones.split == split]
        df_eval = pred_split.merge(
            features[["store_nbr", "family", "week_start", "ventas"]],
            on=["store_nbr", "family", "week_start"], how="left",
        ).rename(columns={col_pred: "prediccion"})
        por_serie = metricas_por_serie(df_eval, df_train_ref)
        r = resumen(por_serie)
        r["cobertura"] = len(por_serie) / features[features.split == split][["store_nbr", "family"]].drop_duplicates().shape[0]
        print(f"  [{nombre}][{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f}")
        fila = {"condicion": nombre, "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)
    return pd.DataFrame(filas_resumen)


def main() -> None:
    resultados_E = pd.read_csv(REPORTS / "resultados_condicion_E_personalizacion.csv")
    test_E = resultados_E[resultados_E.split == "test"].sort_values("wmape_mediana")
    ganadora = test_E.iloc[0]["condicion"]
    nombre_variante = [v for v in m17.VARIANTES if v in ganadora][0]
    cfg = m17.VARIANTES[nombre_variante]
    print(f"Variante de personalización ganadora (Etapa 3): {nombre_variante} -> {cfg}")

    modelo_global, ganador_D = m17.cargar_mejor_modelo_D()
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")

    predicciones = personalizar_con_y_sin_duan(features, modelo_global, cfg["lr"], cfg["congelar_base"])

    resultados = [
        evaluar(predicciones, features, "pred_sin_corregir", f"E - {nombre_variante} (sin corregir)"),
        evaluar(predicciones, features, "pred_corregida", f"E - {nombre_variante} (con Duan)"),
    ]
    df_resumen = pd.concat(resultados, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_correccion_duan.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))

    print(f"\nFactor de Duan -- distribución entre tiendas:")
    print(predicciones.drop_duplicates("store_nbr")["factor_duan"].describe())


if __name__ == "__main__":
    main()
