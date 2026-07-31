"""
12_condicion_a_local.py — T2.3: Condición A (Local).

Un MLP+embeddings (modelo_mlp.py, arquitectura fijada en la Sesión 5) por tienda: cada una
entrena SOLO con sus propios datos, sin ninguna colaboración con las demás. Representa el
escenario "sin federar" -- la referencia de partida que RQ1 pregunta si el federado (D) puede
acercar al techo centralizado (C) sin que las tiendas compartan datos crudos.

A diferencia de la corrección de T2.2b (Sesión 22) -- donde un LightGBM por tienda no lograba
batir a un baseline simple por falta de datos, y se sustituyó por un modelo global -- aquí la
escasez de datos por tienda NO se corrige: es precisamente lo que esta condición debe representar
fielmente. Un local débil por falta de datos es el problema que el resto del experimento (B-E)
está diseñado para demostrar que se puede mitigar.
"""
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import metricas_por_serie, resumen
from modelo_mlp import ArquitecturaMLP, entrenar, predecir

ROOT = Path(__file__).resolve().parents[1]


def entrenar_locales(features: pd.DataFrame, arq: ArquitecturaMLP, lr: float) -> pd.DataFrame:
    """Entrena un MLP independiente por tienda y devuelve las predicciones de val+test."""
    filas = []
    tiendas = sorted(features["store_nbr"].unique())
    print(f"Entrenando MLP local para {len(tiendas)} tiendas...")
    for i, store in enumerate(tiendas, start=1):
        d = features[features.store_nbr == store]
        train = d[d.split == "train"]
        val = d[d.split == "val"]
        eval_ = d[d.split.isin(["val", "test"])]
        if len(train) < 20 or len(eval_) == 0:
            continue

        if len(val) >= 10:
            modelo, hist = entrenar(train, val, arq=arq, lr=lr)
        else:
            # sin val propio (p.ej. tienda 52, apertura tardía -- Sesión 16): se reserva una
            # porción final de train como val interno solo para decidir el early stopping.
            corte = int(len(train) * 0.85)
            train_int, val_int = train.iloc[:corte], train.iloc[corte:]
            if len(val_int) < 5:
                val_int = train_int
            modelo, hist = entrenar(train_int, val_int, arq=arq, lr=lr)

        pred = predecir(modelo, eval_)
        fila = eval_[["store_nbr", "family", "week_start", "split"]].copy()
        fila["pred_local"] = pred
        filas.append(fila)

        if i % 10 == 0:
            print(f"  {i}/{len(tiendas)} tiendas entrenadas (última: {hist['epocas_entrenadas']} épocas, "
                  f"val_loss={hist['mejor_val_loss']:.4f})")

    return pd.concat(filas, ignore_index=True)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed = ROOT / cfg.paths.processed
    reports = ROOT / cfg.paths.reports
    arq = ArquitecturaMLP(dim_emb=cfg.modelo.dim_emb, hidden1=cfg.modelo.hidden1,
                           hidden2=cfg.modelo.hidden2, dropout=cfg.modelo.dropout)

    features = pd.read_parquet(processed / "dataset_features.parquet")
    predicciones = entrenar_locales(features, arq=arq, lr=cfg.modelo.lr)

    df_train_ref = features[features.split == "train"][["store_nbr", "family", "ventas"]]
    filas_resumen = []
    for split in ["val", "test"]:
        pred_split = predicciones[predicciones.split == split]
        df_eval = pred_split.merge(
            features[["store_nbr", "family", "week_start", "ventas"]],
            on=["store_nbr", "family", "week_start"], how="left",
        ).rename(columns={"pred_local": "prediccion"})

        por_serie = metricas_por_serie(df_eval, df_train_ref)
        n_esperado = features[features.split == split][["store_nbr", "family"]].drop_duplicates().shape[0]
        r = resumen(por_serie)
        r["cobertura"] = len(por_serie) / n_esperado
        print(f"[{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f} "
              f"cobertura={r['cobertura']*100:.1f}%")
        fila = {"condicion": "A - Local", "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)

    df_resumen = pd.DataFrame(filas_resumen)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "resultados_condicion_A_local.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))


if __name__ == "__main__":
    main()
