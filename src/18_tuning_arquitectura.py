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

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import wmape
from modelo_mlp import ArquitecturaMLP, entrenar, predecir

ROOT = Path(__file__).resolve().parents[1]


def _candidatos(espacio_busqueda: dict, baseline: dict, n: int, semilla: int) -> list[dict]:
    claves = list(espacio_busqueda.keys())
    candidatos = [dict(baseline)]
    for i in range(n):
        rng = np.random.RandomState(semilla + i)
        candidatos.append({k: rng.choice(espacio_busqueda[k]).item() for k in claves})
    return candidatos


@hydra.main(config_path="../conf", config_name="tuning_arquitectura", version_base=None)
def main(cfg: DictConfig) -> None:
    processed = ROOT / cfg.paths.processed
    configs_dir = ROOT / cfg.paths.configs
    espacio_busqueda = OmegaConf.to_container(cfg.espacio_busqueda, resolve=True)
    baseline = OmegaConf.to_container(cfg.baseline, resolve=True)

    features = pd.read_parquet(processed / "dataset_features.parquet")
    train = features[features.split == "train"]
    val = features[features.split == "val"]

    candidatos = _candidatos(espacio_busqueda, baseline, cfg.n_candidatos, cfg.semilla)
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

    configs_dir.mkdir(parents=True, exist_ok=True)
    salida = {
        "lr": mejor_config["lr"], "batch_size": int(mejor_config["batch_size"]),
        "dim_emb": mejor_arq.dim_emb, "hidden1": mejor_arq.hidden1,
        "hidden2": mejor_arq.hidden2, "dropout": mejor_arq.dropout,
        "wmape_val": mejor_wmape,
    }
    with open(configs_dir / "mlp_arquitectura.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {configs_dir / 'mlp_arquitectura.json'}")

    pd.DataFrame(resultados).to_csv(ROOT / cfg.paths.reports / "tuning_arquitectura_mlp.csv", index=False)


if __name__ == "__main__":
    main()
