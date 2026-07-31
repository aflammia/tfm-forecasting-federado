"""
16_condicion_d_federado.py — T3.2: Condición D (Federado, FedAvg vs. FedProx).

Los 3 silos entrenan de forma federada: el servidor nunca ve las filas de ninguna tienda, solo
los pesos que cada silo devuelve tras `epocas_locales` épocas locales por ronda (McMahan et al.,
2017). Se compara FedAvg con FedProx (Li et al., 2020) -- porque la Sesión 24 (Condición B) ya
encontró heterogeneidad no-IID real entre tiendas de un mismo silo, y FedProx está diseñado
específicamente para ese escenario -- y con distintas `epocas_locales` por ronda.

La selección de la MEJOR ronda usa la pérdida centralizada sobre el val GLOBAL (las 13.992 filas
de las 3 silos juntas) -- no el val de un silo suelto, para evitar el mismo problema de ruido que
ya afectó a la decisión de early stopping en versiones anteriores (Sesión 21/24: un val pequeño
da una señal de parada inestable).

CORRECCIÓN (Sesión 28, tuning): usa la arquitectura y el `lr` ganadores de la búsqueda de la
Etapa 1 (`configs/mlp_arquitectura.json`, `src/18_tuning_arquitectura.py`) en vez de los valores
de fábrica -- ninguna condición A-E había pasado antes por este ajuste (a diferencia de LightGBM,
Sesión 22). Se prueban `epocas_locales ∈ {1, 2, 3}` con FedAvg (la estrategia ganadora de la
Sesión 26); FedProx se repite solo con `epocas_locales=2` (el valor ya usado en su comparación
original) como control de que la conclusión "FedAvg > FedProx" se sostiene con la nueva
arquitectura -- no se repite la búsqueda de μ, fuera de alcance de esta etapa (ver plan de
tuning).
"""
import json
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from federado_flower import MLPConEmbeddings, cargar_mejor_ronda, ejecutar_federado
from metrics import metricas_por_serie, resumen
from modelo_mlp import ArquitecturaMLP, predecir

ROOT = Path(__file__).resolve().parents[1]


def cargar_arquitectura_ganadora(configs_dir: Path) -> tuple[ArquitecturaMLP, float]:
    """Lee configs/mlp_arquitectura.json (Etapa 1). Si no existe todavía, usa la arquitectura de
    fábrica (Sesión 5/22) -- para que el script siga siendo ejecutable de forma independiente."""
    ruta = configs_dir / "mlp_arquitectura.json"
    if not ruta.exists():
        print("Aviso: configs/mlp_arquitectura.json no existe -- usando arquitectura de fábrica.")
        return ArquitecturaMLP(), 1e-3
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    arq = ArquitecturaMLP(dim_emb=cfg["dim_emb"], hidden1=cfg["hidden1"],
                           hidden2=cfg["hidden2"], dropout=cfg["dropout"])
    print(f"Arquitectura cargada de {ruta}: {arq}, lr={cfg['lr']}")
    return arq, cfg["lr"]


def evaluar_modelo(modelo: MLPConEmbeddings, features: pd.DataFrame, nombre: str,
                    carpeta_checkpoint: str) -> pd.DataFrame:
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
        fila = {"condicion": nombre, "split": split, "carpeta_checkpoint": carpeta_checkpoint}
        fila.update(r.to_dict())
        filas_resumen.append(fila)
    return pd.DataFrame(filas_resumen)


@hydra.main(config_path="../conf", config_name="condicion_d", version_base=None)
def main(cfg: DictConfig) -> None:
    processed = ROOT / cfg.paths.processed
    reports = ROOT / cfg.paths.reports
    configs_dir = ROOT / cfg.paths.configs
    num_rounds = cfg.num_rounds
    epocas_locales_a_probar = list(cfg.epocas_locales_a_probar)
    proximal_mu = cfg.proximal_mu

    arq, lr = cargar_arquitectura_ganadora(configs_dir)
    features = pd.read_parquet(processed / "dataset_features.parquet")
    train = features[features.split == "train"]
    val = features[features.split == "val"]

    datos_por_silo = {silo: g for silo, g in train.groupby("silo")}
    for silo, g in datos_por_silo.items():
        print(f"Silo {silo}: {g['store_nbr'].nunique()} tiendas, {len(g)} filas de train")

    resultados = []
    historiales = {}

    for epocas_locales in epocas_locales_a_probar:
        nombre = f"D - FedAvg (epocas_locales={epocas_locales})"
        carpeta_id = f"fedavg_el{epocas_locales}"
        print(f"\n{'=' * 60}\n{nombre} — {num_rounds} rondas\n{'=' * 60}")
        carpeta = processed / "checkpoints_federado" / carpeta_id
        hist = ejecutar_federado(
            datos_por_silo, val, carpeta, estrategia="fedavg", num_rounds=num_rounds,
            epocas_locales=epocas_locales, lr=lr, arq=arq,
        )
        modelo, mejor_ronda = cargar_mejor_ronda(carpeta, hist, arq=arq)
        print(f"Mejor ronda: {mejor_ronda} (val_loss={hist[mejor_ronda]:.4f})")
        resultados.append(evaluar_modelo(modelo, features, nombre, carpeta_id))
        historiales[carpeta_id] = hist

    nombre_fedprox = "D - FedProx (epocas_locales=2, control)"
    carpeta_id_fedprox = "fedprox_el2"
    print(f"\n{'=' * 60}\n{nombre_fedprox} (mu={proximal_mu}) — {num_rounds} rondas\n{'=' * 60}")
    carpeta_fedprox = processed / "checkpoints_federado" / carpeta_id_fedprox
    hist_fedprox = ejecutar_federado(
        datos_por_silo, val, carpeta_fedprox, estrategia="fedprox", proximal_mu=proximal_mu,
        num_rounds=num_rounds, epocas_locales=2, lr=lr, arq=arq,
    )
    modelo_fedprox, ronda_fedprox = cargar_mejor_ronda(carpeta_fedprox, hist_fedprox, arq=arq)
    print(f"Mejor ronda FedProx: {ronda_fedprox} (val_loss={hist_fedprox[ronda_fedprox]:.4f})")
    resultados.append(evaluar_modelo(modelo_fedprox, features, nombre_fedprox, carpeta_id_fedprox))
    historiales[carpeta_id_fedprox] = hist_fedprox

    df_resumen = pd.concat(resultados, ignore_index=True)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "resultados_condicion_D_federado.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))

    test_ordenado = df_resumen[df_resumen.split == "test"].sort_values("wmape_mediana")
    ganador = test_ordenado.iloc[0]["condicion"]
    print(f"\nGanador de la Condición D (menor WMAPE test mediana): {ganador}")

    filas_hist = []
    for nombre_run, hist in historiales.items():
        for ronda, perdida in hist.items():
            filas_hist.append({"config": nombre_run, "ronda": ronda, "val_loss": perdida})
    pd.DataFrame(filas_hist).to_csv(reports / "historial_rondas_condicion_D.csv", index=False)


if __name__ == "__main__":
    main()
