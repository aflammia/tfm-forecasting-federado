"""
16_condicion_d_federado.py — T3.2: Condición D (Federado, FedAvg vs. FedProx).

Los 3 silos entrenan de forma federada: el servidor nunca ve las filas de ninguna tienda, solo
los pesos que cada silo devuelve tras `epocas_locales` épocas locales por ronda (McMahan et al.,
2017). Se ejecutan DOS estrategias -- FedAvg y FedProx -- porque la Sesión 24 (Condición B) ya
encontró heterogeneidad no-IID real entre tiendas de un mismo silo, y FedProx (Li et al., 2020)
está diseñado específicamente para ese escenario (penaliza que los pesos locales se alejen
demasiado de los globales en cada ronda). Se reporta la que gane, con el resultado de la otra
como referencia.

La selección de la MEJOR ronda usa la pérdida centralizada sobre el val GLOBAL (las 13.992 filas
de las 3 silos juntas) -- no el val de un silo suelto, para evitar el mismo problema de ruido que
ya afectó a la decisión de early stopping en versiones anteriores (Sesión 21/24: un val pequeño
da una señal de parada inestable).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import FEATURES_CONTINUAS, predecir
from federado_flower import ejecutar_federado, cargar_mejor_ronda, set_params, MLPConEmbeddings
from metrics import metricas_por_serie, resumen

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"

NUM_ROUNDS = 15  # calibrado empíricamente: con 2 épocas locales/ronda, val_loss ya converge hacia
                 # la ronda 3 y empieza a subir levemente hacia la 5 (sobreajuste) -- 15 da margen
                 # de sobra para confirmar el mejor punto sin gastar cómputo innecesario (~40s/ronda)
EPOCAS_LOCALES = 2
PROXIMAL_MU = 0.01


def evaluar_modelo(modelo: MLPConEmbeddings, features: pd.DataFrame, nombre: str) -> pd.DataFrame:
    df_train_ref = features[features.split == "train"][["store_nbr", "family", "ventas"]]
    filas_resumen = []
    for split in ["val", "test"]:
        d = features[features.split == split]
        pred = predecir(modelo, d)
        df_eval = d[["store_nbr", "family", "ventas"]].copy()
        df_eval["prediccion"] = pred
        por_serie = metricas_por_serie(df_eval, df_train_ref)
        r = resumen(por_serie)
        r["cobertura"] = len(por_serie) / d[["store_nbr", "family"]].drop_duplicates().shape[0]
        print(f"  [{nombre}][{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f} "
              f"cobertura={r['cobertura']*100:.1f}%")
        fila = {"condicion": nombre, "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)
    return pd.DataFrame(filas_resumen)


def main() -> None:
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    train = features[features.split == "train"]
    val = features[features.split == "val"]

    datos_por_silo = {silo: g for silo, g in train.groupby("silo")}
    for silo, g in datos_por_silo.items():
        print(f"Silo {silo}: {g['store_nbr'].nunique()} tiendas, {len(g)} filas de train")

    resultados = []

    print(f"\n{'=' * 60}\nFedAvg — {NUM_ROUNDS} rondas, {EPOCAS_LOCALES} épocas locales/ronda\n{'=' * 60}")
    hist_fedavg = ejecutar_federado(
        datos_por_silo, val, PROCESSED / "checkpoints_federado" / "fedavg",
        estrategia="fedavg", num_rounds=NUM_ROUNDS, epocas_locales=EPOCAS_LOCALES,
    )
    modelo_fedavg, ronda_fedavg = cargar_mejor_ronda(PROCESSED / "checkpoints_federado" / "fedavg", hist_fedavg)
    print(f"Mejor ronda FedAvg: {ronda_fedavg} (val_loss={hist_fedavg[ronda_fedavg]:.4f})")
    resultados.append(evaluar_modelo(modelo_fedavg, features, "D - FedAvg"))

    print(f"\n{'=' * 60}\nFedProx (mu={PROXIMAL_MU}) — {NUM_ROUNDS} rondas, {EPOCAS_LOCALES} épocas locales/ronda\n{'=' * 60}")
    hist_fedprox = ejecutar_federado(
        datos_por_silo, val, PROCESSED / "checkpoints_federado" / "fedprox",
        estrategia="fedprox", proximal_mu=PROXIMAL_MU, num_rounds=NUM_ROUNDS, epocas_locales=EPOCAS_LOCALES,
    )
    modelo_fedprox, ronda_fedprox = cargar_mejor_ronda(PROCESSED / "checkpoints_federado" / "fedprox", hist_fedprox)
    print(f"Mejor ronda FedProx: {ronda_fedprox} (val_loss={hist_fedprox[ronda_fedprox]:.4f})")
    resultados.append(evaluar_modelo(modelo_fedprox, features, "D - FedProx"))

    df_resumen = pd.concat(resultados, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_condicion_D_federado.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))

    # historiales de pérdida por ronda, para inspeccionar convergencia
    pd.DataFrame({"ronda": list(hist_fedavg.keys()), "val_loss_fedavg": list(hist_fedavg.values())}) \
        .merge(pd.DataFrame({"ronda": list(hist_fedprox.keys()), "val_loss_fedprox": list(hist_fedprox.values())}), on="ronda") \
        .to_csv(REPORTS / "historial_rondas_condicion_D.csv", index=False)


if __name__ == "__main__":
    main()
