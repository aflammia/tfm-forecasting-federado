"""
13_condicion_b_silo.py — T2.4: Condición B (Centralizado por silo).

Un MLP+embeddings (modelo_mlp.py) por SILO (3 modelos: Grande, Mediano, Pequeño), entrenado con
las filas de TODAS las tiendas de ese silo juntas -- simula que las tiendas de un mismo operador
retail centralizan sus datos internamente (legítimo, es el mismo operador simulado; Sesión 5).

Es el nivel intermedio entre A (Local, sin colaboración) y C (Centralizado global, los 3 silos
juntos): si B ya mejora sobre A, confirma que agrupar datos ayuda -- la pregunta de fondo que D
(Federado) responde es si se puede conseguir una mejora parecida SIN que los silos compartan
datos crudos entre ellos.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import entrenar, predecir
from metrics import metricas_por_serie, resumen

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def entrenar_por_silo(features: pd.DataFrame) -> pd.DataFrame:
    filas = []
    silos = sorted(features["silo"].unique())
    print(f"Entrenando MLP centralizado para {len(silos)} silos...")
    for silo in silos:
        d = features[features.silo == silo]
        train = d[d.split == "train"]
        val = d[d.split == "val"]
        eval_ = d[d.split.isin(["val", "test"])]
        print(f"  Silo {silo}: {d['store_nbr'].nunique()} tiendas, {len(train)} filas de train, {len(val)} de val")

        modelo, hist = entrenar(train, val)
        pred = predecir(modelo, eval_)
        fila = eval_[["store_nbr", "family", "week_start", "split"]].copy()
        fila["pred_silo"] = pred
        filas.append(fila)
        print(f"    {hist['epocas_entrenadas']} épocas, mejor val_loss={hist['mejor_val_loss']:.4f}")

    return pd.concat(filas, ignore_index=True)


def main() -> None:
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    predicciones = entrenar_por_silo(features)

    df_train_ref = features[features.split == "train"][["store_nbr", "family", "ventas"]]
    filas_resumen = []
    for split in ["val", "test"]:
        pred_split = predicciones[predicciones.split == split]
        df_eval = pred_split.merge(
            features[["store_nbr", "family", "week_start", "ventas"]],
            on=["store_nbr", "family", "week_start"], how="left",
        ).rename(columns={"pred_silo": "prediccion"})

        por_serie = metricas_por_serie(df_eval, df_train_ref)
        n_esperado = features[features.split == split][["store_nbr", "family"]].drop_duplicates().shape[0]
        r = resumen(por_serie)
        r["cobertura"] = len(por_serie) / n_esperado
        print(f"[{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f} "
              f"cobertura={r['cobertura']*100:.1f}%")
        fila = {"condicion": "B - Centralizado por silo", "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)

    df_resumen = pd.DataFrame(filas_resumen)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_condicion_B_silo.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))


if __name__ == "__main__":
    main()
