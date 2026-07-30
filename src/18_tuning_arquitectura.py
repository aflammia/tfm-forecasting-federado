"""
18_tuning_arquitectura.py — Etapa 1 del plan de tuning (Sesión 28): arquitectura y optimización
del MLP.

No usa Flower -- es un proxy rápido, el mismo patrón que la búsqueda de hiperparámetros de
LightGBM (Sesión 22): un único `entrenar()` sobre el dataset GLOBAL agrupado (todo train, todo
val), seleccionando por WMAPE en escala natural sobre val (no por la pérdida de entrenamiento en
log). Objetivo: encontrar una arquitectura/optimización razonable para llevar después a la
Condición D (federado, Etapa 2) y E (personalización, Etapa 3) -- ninguna de las condiciones A-E
había pasado antes por este ajuste, a diferencia de LightGBM (que por eso ganó, Sesión 22).

El primer candidato es siempre la arquitectura ORIGINAL (Sesión 5/22: 33→64→32→1, dropout 0,2,
embeddings de 8 dim, lr=1e-3, batch=256) -- para tener una comparación limpia de "antes/después"
frente a los candidatos de la búsqueda aleatoria.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import ArquitecturaMLP, entrenar, predecir
from metrics import wmape

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

ESPACIO_BUSQUEDA = {
    "lr": [5e-4, 1e-3, 2e-3],
    "dropout": [0.1, 0.2, 0.3],
    "hidden1": [64, 96, 128],
    "dim_emb": [8, 16],
    "batch_size": [128, 256],
}

BASELINE = {"lr": 1e-3, "dropout": 0.2, "hidden1": 64, "dim_emb": 8, "batch_size": 256}


def _candidatos(n: int, semilla: int) -> list[dict]:
    claves = list(ESPACIO_BUSQUEDA.keys())
    candidatos = [dict(BASELINE)]
    for i in range(n):
        rng = np.random.RandomState(semilla + i)
        candidatos.append({k: rng.choice(ESPACIO_BUSQUEDA[k]).item() for k in claves})
    return candidatos


def main(n_candidatos: int = 12, semilla: int = 42) -> None:
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    train = features[features.split == "train"]
    val = features[features.split == "val"]

    candidatos = _candidatos(n_candidatos, semilla)
    resultados = []
    mejor_config, mejor_wmape, mejor_arq = None, np.inf, None

    print(f"Buscando arquitectura/optimización del MLP ({len(candidatos)} candidatos, "
          f"el 1º es la arquitectura original)...")
    for i, config in enumerate(candidatos, start=1):
        arq = ArquitecturaMLP(dim_emb=int(config["dim_emb"]), hidden1=int(config["hidden1"]),
                               hidden2=int(config["hidden1"]) // 2, dropout=float(config["dropout"]))
        modelo, hist = entrenar(
            train, val, lr=float(config["lr"]), batch_size=int(config["batch_size"]),
            arq=arq, epochs=100, paciencia=12, semilla=42,
        )
        pred_val = predecir(modelo, val)
        w = wmape(val["ventas"].values, pred_val)
        etiqueta = "ORIGINAL" if i == 1 else f"candidato {i - 1}"
        print(f"  [{etiqueta}] lr={config['lr']} dropout={config['dropout']} "
              f"hidden1={config['hidden1']} dim_emb={config['dim_emb']} batch={config['batch_size']} "
              f"-> WMAPE val={w:.4f} ({hist['epocas_entrenadas']} épocas)")
        resultados.append({**config, "wmape_val": w, "epocas": hist["epocas_entrenadas"], "etiqueta": etiqueta})
        if w < mejor_wmape:
            mejor_wmape, mejor_config, mejor_arq = w, config, arq

    print(f"\nMejor configuración (WMAPE val={mejor_wmape:.4f}): {mejor_config}")
    mejora_pct = (resultados[0]["wmape_val"] - mejor_wmape) / resultados[0]["wmape_val"] * 100
    print(f"Mejora sobre la arquitectura original (WMAPE val {resultados[0]['wmape_val']:.4f}): "
          f"{mejora_pct:+.1f}%")

    CONFIGS.mkdir(parents=True, exist_ok=True)
    salida = {
        "lr": mejor_config["lr"], "batch_size": int(mejor_config["batch_size"]),
        "dim_emb": mejor_arq.dim_emb, "hidden1": mejor_arq.hidden1,
        "hidden2": mejor_arq.hidden2, "dropout": mejor_arq.dropout,
        "wmape_val": mejor_wmape,
    }
    with open(CONFIGS / "mlp_arquitectura.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {CONFIGS / 'mlp_arquitectura.json'}")

    pd.DataFrame(resultados).to_csv(
        Path(__file__).resolve().parents[1] / "reports" / "tuning_arquitectura_mlp.csv", index=False
    )


if __name__ == "__main__":
    main()
