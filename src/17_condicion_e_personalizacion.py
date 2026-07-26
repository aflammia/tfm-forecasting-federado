"""
17_condicion_e_personalizacion.py — T3.3: Condición E (Federado + personalización).

Parte del modelo GLOBAL ya convergido de la Condición D (la mejor entre FedAvg/FedProx, según
`reports/resultados_condicion_D_federado.csv`) y hace un fine-tuning corto por tienda individual
(estilo FedPer/personalización post-federado) -- no vuelve a entrenar desde cero como A, continúa
desde los pesos que ya capturaron la estructura compartida entre silos.

Es la condición que más directamente responde a lo encontrado en la Sesión 24 (B y C, sin
mecanismo de personalización, rindieron PEOR que A por heterogeneidad no-IID entre tiendas):
la hipótesis es que partir de un buen punto de partida global y luego especializar por tienda
recupera lo mejor de ambos mundos -- la señal compartida del federado (D) y la especialización
del local (A), sin el coste de partir de cero por tienda.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import entrenar, predecir, MLPConEmbeddings
from federado_flower import set_params
from metrics import metricas_por_serie, resumen

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def cargar_mejor_modelo_D() -> tuple[MLPConEmbeddings, str]:
    resultados_D = pd.read_csv(REPORTS / "resultados_condicion_D_federado.csv")
    test_D = resultados_D[resultados_D.split == "test"].sort_values("wmape_mediana")
    ganador = test_D.iloc[0]["condicion"]
    print(f"Condición D ganadora (menor WMAPE test mediana): {ganador}")

    carpeta = "fedavg" if "FedAvg" in ganador else "fedprox"
    hist_path = REPORTS / "historial_rondas_condicion_D.csv"
    hist_df = pd.read_csv(hist_path)
    col_perdida = "val_loss_fedavg" if carpeta == "fedavg" else "val_loss_fedprox"
    mejor_ronda = int(hist_df.loc[hist_df[col_perdida].idxmin(), "ronda"])

    checkpoint = PROCESSED / "checkpoints_federado" / carpeta / f"ronda_{mejor_ronda}.npz"
    datos = np.load(checkpoint)
    parametros = [datos[k] for k in datos.files]
    modelo = MLPConEmbeddings()
    set_params(modelo, parametros)
    return modelo, ganador


def personalizar_por_tienda(features: pd.DataFrame, modelo_global: MLPConEmbeddings) -> pd.DataFrame:
    filas = []
    tiendas = sorted(features["store_nbr"].unique())
    print(f"Personalizando (fine-tuning) para {len(tiendas)} tiendas, partiendo del modelo global...")
    for i, store in enumerate(tiendas, start=1):
        d = features[features.store_nbr == store]
        train = d[d.split == "train"]
        val = d[d.split == "val"]
        eval_ = d[d.split.isin(["val", "test"])]
        if len(train) < 20 or len(eval_) == 0:
            continue

        if len(val) >= 10:
            modelo, hist = entrenar(train, val, epochs=50, paciencia=10, lr=5e-4,
                                     modelo_inicial=modelo_global)
        else:
            corte = int(len(train) * 0.85)
            train_int, val_int = train.iloc[:corte], train.iloc[corte:]
            if len(val_int) < 5:
                val_int = train_int
            modelo, hist = entrenar(train_int, val_int, epochs=50, paciencia=10, lr=5e-4,
                                     modelo_inicial=modelo_global)

        pred = predecir(modelo, eval_)
        fila = eval_[["store_nbr", "family", "week_start", "split"]].copy()
        fila["pred_personalizado"] = pred
        filas.append(fila)

        if i % 10 == 0:
            print(f"  {i}/{len(tiendas)} tiendas ({hist['epocas_entrenadas']} épocas, "
                  f"val_loss={hist['mejor_val_loss']:.4f})")

    return pd.concat(filas, ignore_index=True)


def main() -> None:
    modelo_global, ganador_D = cargar_mejor_modelo_D()
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    predicciones = personalizar_por_tienda(features, modelo_global)

    df_train_ref = features[features.split == "train"][["store_nbr", "family", "ventas"]]
    filas_resumen = []
    for split in ["val", "test"]:
        pred_split = predicciones[predicciones.split == split]
        df_eval = pred_split.merge(
            features[["store_nbr", "family", "week_start", "ventas"]],
            on=["store_nbr", "family", "week_start"], how="left",
        ).rename(columns={"pred_personalizado": "prediccion"})

        por_serie = metricas_por_serie(df_eval, df_train_ref)
        n_esperado = features[features.split == split][["store_nbr", "family"]].drop_duplicates().shape[0]
        r = resumen(por_serie)
        r["cobertura"] = len(por_serie) / n_esperado
        print(f"[{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f} "
              f"cobertura={r['cobertura']*100:.1f}%")
        fila = {"condicion": f"E - Federado + personalización (desde {ganador_D})", "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)

        por_serie.to_csv(REPORTS / f"por_serie_condicion_E_{split}.csv", index=False)

    df_resumen = pd.DataFrame(filas_resumen)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_condicion_E_personalizacion.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))


if __name__ == "__main__":
    main()
