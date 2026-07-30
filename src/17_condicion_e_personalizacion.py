"""
17_condicion_e_personalizacion.py — T3.3: Condición E (Federado + personalización).

Parte del modelo GLOBAL ya convergido de la Condición D (la mejor configuración según
`reports/resultados_condicion_D_federado.csv`) y hace un fine-tuning corto por tienda individual
(personalización post-federado) -- no vuelve a entrenar desde cero como A, continúa desde los
pesos que ya capturaron la estructura compartida entre silos.

Es la condición que más directamente responde a lo encontrado en la Sesión 24 (B y C, sin
mecanismo de personalización, rindieron PEOR que A por heterogeneidad no-IID entre tiendas):
la hipótesis es que partir de un buen punto de partida global y luego especializar por tienda
recupera lo mejor de ambos mundos -- la señal compartida del federado (D) y la especialización
del local (A), sin el coste de partir de cero por tienda.

CORRECCIÓN (Sesión 28, tuning): se comparan tres variantes de personalización (Etapa 3 del plan
de tuning):
  1. "completo" — la original (Sesión 27): fine-tuning de TODOS los pesos, lr=5e-4.
  2. "lr_bajo" — fine-tuning completo con un lr más suave (2e-4), para no deshacer de golpe lo
     aprendido de forma federada.
  3. "fedper" — estilo FedPer (Arivazhagan et al., 2019): se congelan las capas densas
     compartidas (`congelar_base=True` en `entrenar()`) y solo se afinan los embeddings y la
     capa de salida -- la apuesta de FedPer es que la "base" generaliza bien y re-entrenarla por
     tienda con pocos datos solo añade varianza.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_mlp import ArquitecturaMLP, entrenar, predecir, MLPConEmbeddings
from federado_flower import set_params
from metrics import metricas_por_serie, resumen

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

VARIANTES = {
    "completo": {"lr": 5e-4, "congelar_base": False},
    "lr_bajo": {"lr": 2e-4, "congelar_base": False},
    "fedper": {"lr": 5e-4, "congelar_base": True},
}


def _cargar_arquitectura() -> ArquitecturaMLP:
    ruta = CONFIGS / "mlp_arquitectura.json"
    if not ruta.exists():
        return ArquitecturaMLP()
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    return ArquitecturaMLP(dim_emb=cfg["dim_emb"], hidden1=cfg["hidden1"],
                            hidden2=cfg["hidden2"], dropout=cfg["dropout"])


def cargar_mejor_modelo_D() -> tuple[MLPConEmbeddings, str]:
    resultados_D = pd.read_csv(REPORTS / "resultados_condicion_D_federado.csv")
    test_D = resultados_D[resultados_D.split == "test"].sort_values("wmape_mediana")
    fila_ganadora = test_D.iloc[0]
    ganador, carpeta_id = fila_ganadora["condicion"], fila_ganadora["carpeta_checkpoint"]
    print(f"Condición D ganadora (menor WMAPE test mediana): {ganador} -> checkpoints/{carpeta_id}")

    hist_df = pd.read_csv(REPORTS / "historial_rondas_condicion_D.csv")
    hist_de_esta_config = hist_df[hist_df.config == carpeta_id]
    mejor_ronda = int(hist_de_esta_config.loc[hist_de_esta_config.val_loss.idxmin(), "ronda"])

    arq = _cargar_arquitectura()
    checkpoint = PROCESSED / "checkpoints_federado" / carpeta_id / f"ronda_{mejor_ronda}.npz"
    datos = np.load(checkpoint)
    parametros = [datos[k] for k in datos.files]
    modelo = MLPConEmbeddings(arq=arq)
    set_params(modelo, parametros)
    return modelo, ganador


def personalizar_por_tienda(features: pd.DataFrame, modelo_global: MLPConEmbeddings,
                             lr: float, congelar_base: bool) -> pd.DataFrame:
    filas = []
    tiendas = sorted(features["store_nbr"].unique())
    print(f"Personalizando (fine-tuning, lr={lr}, congelar_base={congelar_base}) para "
          f"{len(tiendas)} tiendas...")
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

        pred = predecir(modelo, eval_)
        fila = eval_[["store_nbr", "family", "week_start", "split"]].copy()
        fila["pred_personalizado"] = pred
        filas.append(fila)

        if i % 15 == 0:
            print(f"  {i}/{len(tiendas)} tiendas ({hist['epocas_entrenadas']} épocas, "
                  f"val_loss={hist['mejor_val_loss']:.4f})")

    return pd.concat(filas, ignore_index=True)


def evaluar(predicciones: pd.DataFrame, features: pd.DataFrame, nombre: str) -> pd.DataFrame:
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
        print(f"  [{nombre}][{split}] WMAPE media={r['wmape_media']:.4f} mediana={r['wmape_mediana']:.4f} "
              f"MASE mediana={r['mase_mediana']:.4f} RMSSE mediana={r['rmsse_mediana']:.4f} "
              f"cobertura={r['cobertura']*100:.1f}%")
        fila = {"condicion": nombre, "split": split}
        fila.update(r.to_dict())
        filas_resumen.append(fila)

        if split == "test":
            por_serie.to_csv(REPORTS / f"por_serie_condicion_E_{nombre}_{split}.csv", index=False)
    return pd.DataFrame(filas_resumen)


def main() -> None:
    modelo_global, ganador_D = cargar_mejor_modelo_D()
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")

    resultados = []
    for nombre_variante, cfg in VARIANTES.items():
        print(f"\n{'=' * 60}\nVariante de personalización: {nombre_variante}\n{'=' * 60}")
        predicciones = personalizar_por_tienda(features, modelo_global, cfg["lr"], cfg["congelar_base"])
        etiqueta = f"E - {nombre_variante} (desde {ganador_D})"
        resultados.append(evaluar(predicciones, features, etiqueta))

    df_resumen = pd.concat(resultados, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_condicion_E_personalizacion.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_resumen.to_string(index=False))

    test_ordenado = df_resumen[df_resumen.split == "test"].sort_values("wmape_mediana")
    print(f"\nMejor variante de personalización (menor WMAPE test mediana): "
          f"{test_ordenado.iloc[0]['condicion']}")


if __name__ == "__main__":
    main()
