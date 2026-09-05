"""
22_significancia.py — T3.5: contraste estadístico formal entre condiciones y baselines.

Cierra la única pieza que faltaba del diseño experimental (Sesión 1 del RESEARCH_LOG): hasta ahora
las condiciones se comparaban por sus medianas agregadas, sin test de significación, porque solo la
Condición E persistía métricas por serie.

Este script es la ÚNICA fuente de verdad del contraste: calcula (o carga) las métricas por serie de
todos los métodos EN LA MISMA CORRIDA y sobre el mismo universo de series, que es lo que exige un
test pareado, y después ejecuta los contrastes de Wilcoxon con `comparar_condiciones()`
(`src/metrics.py`, ya testeado). No reejecuta lo caro: reutiliza el checkpoint federado ya en disco
y los hiperparámetros de LightGBM ya buscados (`configs/lightgbm_hiperparametros.json`), y no
recalcula ETS (mucho más lento y claramente peor, no aporta al contraste).

Linaje coherente con el resultado titular del TFM: D es el FedAvg ORIGINAL (Sesión 26), que es del
que se derivó la Condición E titular (Sesión 27) -- así la cadena A→B→C→D→E es comparable. La
variante tuneada de D (Sesión 28) se reporta aparte en la memoria.

Salidas: reports/por_serie_<metodo>_test.csv (uno por método) y reports/significancia_wilcoxon.csv

Ejecutar: PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/22_significancia.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from federado_flower import set_params  # noqa: E402
from metrics import comparar_condiciones, metricas_por_serie, resumen  # noqa: E402
from modelo_mlp import (  # noqa: E402
    ArquitecturaMLP,
    DatasetVentas,
    MLPConEmbeddings,
    entrenar,
    predecir,
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

m10 = importlib.import_module("10_baselines_ingenuos")
m11 = importlib.import_module("11_baselines_convencionales")
m12 = importlib.import_module("12_condicion_a_local")
m13 = importlib.import_module("13_condicion_b_silo")

# Arquitectura de fábrica (Sesión 5/22): es la que produjo los resultados titulares de A, B, C, D y E.
ARQ = ArquitecturaMLP()
LR = 1e-3

# Fichero por-serie de la Condición E titular, ya generado en la Sesión 28 al regenerar el
# resultado original (WMAPE test mediana = 0,1379).
E_TITULAR = "por_serie_condicion_E_E - completo (desde D - FedAvg original, Sesion 27)_test.csv"


def _por_serie(pred_df: pd.DataFrame, features: pd.DataFrame, col_pred: str) -> pd.DataFrame:
    """Métricas por serie sobre TEST. `pred_df` debe traer store_nbr, family, week_start, split y
    la columna de predicción. El train de cada serie es la referencia de escala de MASE/RMSSE."""
    df_train_ref = features[features.split == "train"][["store_nbr", "family", "ventas"]]
    d = pred_df[pred_df.split == "test"]
    # Algunas fuentes (p.ej. construir_predicciones de los baselines) ya traen `ventas`; en ese caso
    # no se vuelve a unir, o pandas la duplicaria como ventas_x/ventas_y y metricas_por_serie fallaria.
    if "ventas" not in d.columns:
        d = d.merge(
            features[["store_nbr", "family", "week_start", "ventas"]],
            on=["store_nbr", "family", "week_start"], how="left",
        )
    df_eval = d.dropna(subset=[col_pred]).rename(columns={col_pred: "prediccion"})
    return metricas_por_serie(df_eval, df_train_ref)


def _predicciones_de_modelo(modelo: MLPConEmbeddings, features: pd.DataFrame, col: str) -> pd.DataFrame:
    """Predice con un MLP ya entrenado sobre val+test y devuelve el formato estándar."""
    eval_ = features[features.split.isin(["val", "test"])]
    out = eval_[["store_nbr", "family", "week_start", "split"]].copy()
    out[col] = predecir(modelo, eval_)
    return out


def _mejor_checkpoint_d(features: pd.DataFrame) -> tuple[MLPConEmbeddings, int]:
    """Reconstruye el mejor modelo de la Condición D ORIGINAL (FedAvg, Sesión 26) desde los
    checkpoints en disco, eligiendo la ronda de menor pérdida sobre el val global. No reentrena:
    el historial por ronda de esa corrida se sobrescribió en la Sesión 28, así que la mejor ronda
    se vuelve a determinar aquí evaluando cada checkpoint (mismo criterio que entonces)."""
    carpeta = PROCESSED / "checkpoints_federado" / "fedavg"
    val = features[features.split == "val"]
    dl_val = DataLoader(DatasetVentas(val), batch_size=2048, shuffle=False)
    perdida_fn = nn.HuberLoss()
    modelo = MLPConEmbeddings(arq=ARQ)
    mejor = None
    for ckpt in sorted(carpeta.glob("ronda_*.npz")):
        datos = np.load(ckpt)
        set_params(modelo, [datos[k] for k in datos.files])
        modelo.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x_cont, fam, tienda, y in dl_val:
                total += perdida_fn(modelo(x_cont, fam, tienda), y).item() * len(y)
                n += len(y)
        val_loss = total / n
        if mejor is None or val_loss < mejor[1]:
            mejor = (int(ckpt.stem.split("_")[1]), val_loss)
    ronda = mejor[0]
    datos = np.load(carpeta / f"ronda_{ronda}.npz")
    set_params(modelo, [datos[k] for k in datos.files])
    print(f"  D original: mejor ronda={ronda} (val_loss={mejor[1]:.5f})")
    return modelo, ronda


def calcular_por_serie(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Métricas por serie (test) de cada método. Devuelve {slug: DataFrame}."""
    metodos: dict[str, pd.DataFrame] = {}

    print("[1/7] Media móvil (4 sem.) -- baseline más fuerte de T2.2")
    datos_baselines = m10.construir_predicciones(features.copy())
    metodos["media_movil"] = _por_serie(datos_baselines, features, "pred_media_movil_4")

    print("[2/7] LightGBM global -- con los hiperparámetros ya buscados (Sesión 22)")
    config_lgbm = json.loads((CONFIGS / "lightgbm_hiperparametros.json").read_text(encoding="utf-8"))
    metodos["lightgbm"] = _por_serie(
        m11.predicciones_lightgbm(features, config_lgbm), features, "pred_lightgbm"
    )

    print("[3/7] Condición A (Local) -- 54 modelos, uno por tienda")
    metodos["condicion_A"] = _por_serie(
        m12.entrenar_locales(features, ARQ, LR), features, "pred_local"
    )

    print("[4/7] Condición B (Centralizado por silo)")
    metodos["condicion_B"] = _por_serie(
        m13.entrenar_por_silo(features, ARQ, LR), features, "pred_silo"
    )

    print("[5/7] Condición C (Centralizado global)")
    modelo_c, _ = entrenar(features[features.split == "train"], features[features.split == "val"],
                           arq=ARQ, lr=LR)
    metodos["condicion_C"] = _por_serie(
        _predicciones_de_modelo(modelo_c, features, "pred_global"), features, "pred_global"
    )

    print("[6/7] Condición D (Federado FedAvg original) -- desde checkpoint, sin reentrenar")
    modelo_d, _ = _mejor_checkpoint_d(features)
    metodos["condicion_D"] = _por_serie(
        _predicciones_de_modelo(modelo_d, features, "pred_federado"), features, "pred_federado"
    )

    print("[7/7] Condición E (Federado + personalización) -- se carga la ya calculada (Sesión 28)")
    metodos["condicion_E"] = pd.read_csv(REPORTS / E_TITULAR)

    return metodos


# Contrastes de interés. Los cuatro primeros son los del diseño experimental (RESEARCH_LOG Sesión 1);
# los dos siguientes sostienen el hallazgo de que centralizar empeora; los dos últimos sitúan al
# federado frente a los mejores métodos no federados (están a muy poca distancia en mediana y
# conviene saber si esa distancia es señal o ruido).
CONTRASTES = [
    ("condicion_A", "condicion_D", "A (Local) vs D (Federado)"),
    ("condicion_A", "condicion_E", "A (Local) vs E (Federado + personalizacion)"),
    ("condicion_C", "condicion_D", "C (Centralizado global) vs D (Federado)"),
    ("condicion_D", "condicion_E", "D (Federado) vs E (Federado + personalizacion)"),
    ("condicion_A", "condicion_B", "A (Local) vs B (Centralizado por silo)"),
    ("condicion_A", "condicion_C", "A (Local) vs C (Centralizado global)"),
    ("condicion_E", "media_movil", "E (Federado + personalizacion) vs Media movil"),
    ("condicion_E", "lightgbm", "E (Federado + personalizacion) vs LightGBM global"),
]


def main() -> None:
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    REPORTS.mkdir(parents=True, exist_ok=True)

    metodos = calcular_por_serie(features)

    print("\n" + "=" * 78)
    print("Medianas de WMAPE por metodo (test) -- deben coincidir con lo ya documentado")
    print("=" * 78)
    for slug, df in metodos.items():
        df.to_csv(REPORTS / f"por_serie_{slug}_test.csv", index=False)
        r = resumen(df)
        print(f"  {slug:<14} n={len(df):>5}  WMAPE mediana={r['wmape_mediana']:.4f}")

    print("\n" + "=" * 78)
    print("Contraste de Wilcoxon pareado por serie (WMAPE, test)")
    print("=" * 78)
    filas = []
    for slug_a, slug_b, etiqueta in CONTRASTES:
        res = comparar_condiciones(metodos[slug_a], metodos[slug_b], metrica="wmape")
        mejor = slug_a if res.mediana_a < res.mediana_b else slug_b
        filas.append({
            "contraste": etiqueta, "metodo_a": slug_a, "metodo_b": slug_b,
            "n_series": res.n_series, "mediana_a": res.mediana_a, "mediana_b": res.mediana_b,
            "estadistico": res.estadistico, "p_valor": res.p_valor,
            "significativo_05": res.significativo_05, "mejor": mejor,
        })
        marca = "SIGNIFICATIVO" if res.significativo_05 else "no significativo"
        print(f"  {etiqueta:<52} n={res.n_series:>5}  p={res.p_valor:.3e}  {marca}")

    df_sig = pd.DataFrame(filas)

    # Corrección por comparaciones múltiples: se ejecutan 8 contrastes sobre los mismos datos, así
    # que el p-valor crudo sobreestima la significación. Holm-Bonferroni es el ajuste estándar y
    # menos conservador que Bonferroni simple.
    from statsmodels.stats.multitest import multipletests
    rechaza, p_holm, _, _ = multipletests(df_sig["p_valor"].values, alpha=0.05, method="holm")
    df_sig["p_valor_holm"] = p_holm
    df_sig["significativo_05_holm"] = rechaza

    out = REPORTS / "significancia_wilcoxon.csv"
    df_sig.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(df_sig[["contraste", "n_series", "mediana_a", "mediana_b", "p_valor",
                  "p_valor_holm", "significativo_05_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
