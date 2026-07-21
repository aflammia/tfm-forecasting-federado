"""
14_condicion_c_global.py — T2.5: Condición C (Centralizado global).

Un único MLP+embeddings (modelo_mlp.py), entrenado con TODAS las tiendas de los 3 silos juntas.
Es el "techo" del experimento: en la práctica inviable (exigiría que operadores retail
competidores compartieran sus datos crudos entre sí, Sesión 7), pero es la referencia superior
frente a la que se mide cuánta brecha recupera el federado (D) sin necesitar ese dato compartido
-- la métrica estrella del proyecto (T2.1, `porcentaje_brecha_recuperada`).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import entrenar, predecir
from metrics import metricas_por_serie, resumen

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    train = features[features.split == "train"]
    val = features[features.split == "val"]
    eval_ = features[features.split.isin(["val", "test"])]

    print(f"Entrenando MLP centralizado global: {features['store_nbr'].nunique()} tiendas, "
          f"{len(train)} filas de train, {len(val)} de val")
    modelo, hist = entrenar(train, val)
    print(f"  {hist['epocas_entrenadas']} épocas, mejor val_loss={hist['mejor_val_loss']:.4f}")

    pred = predecir(modelo, eval_)
    predicciones = eval_[["store_nbr", "family", "week_start", "split"]].copy()
    predicciones["pred_global"] = pred

    df_train_ref = train[["store_nbr", "family", "ventas"]]
    filas_resumen = []
    for split in ["val", "test"]:
        pred_split = predicciones[predicciones.split == split]
        df_eval = pred_split.merge(
            features[["store_nbr", "family", "week_start", "ventas"]],
            on=["store_nbr", "family", "week_start"], how="left",
        ).rename(columns={"pred_global": "prediccion"})

        por_serie = metricas_por_serie(df_eval, df_train_ref)
        n_esperado = features[features.split == split][["store_nbr", "family"]].drop_duplicates().shape[0]
        r = resumen(por_serie)
        r["cobertura"] = len(por_serie) / n_esperado
        print(f"[{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f} "
              f"cobertura={r['cobertura']*100:.1f}%")
        fila = {"condicion": "C - Centralizado global", "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)

    df_resumen = pd.DataFrame(filas_resumen)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_condicion_C_global.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))


if __name__ == "__main__":
    main()
