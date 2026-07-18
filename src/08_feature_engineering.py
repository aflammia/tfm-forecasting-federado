"""
08_feature_engineering.py — T1.4: ingeniería de variables.

Añade sobre dataset_modelado.parquet (T1.2+T1.3):
  - Variables autorregresivas: lags (1,2,4,8 semanas), medias moviles (4,8 sem.), desviacion (4 sem.)
  - Objetivo transformado: log(1+ventas) (justificado en RESEARCH_LOG Sesion 5)
  - Codificacion de categoricas para embeddings: family_id, store_id

Todas las variables autorregresivas se calculan POR SERIE (tienda x familia), ordenadas en el
tiempo, usando exclusivamente semanas ANTERIORES a la que se predice (shift antes de rolling) —
se verifica explicitamente que ningun lag usa el futuro.

Salida: data/processed/dataset_features.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

LAGS = [1, 2, 4, 8]
VENTANAS_MEDIA = [4, 8]
VENTANA_STD = 4


def main() -> None:
    data = pd.read_parquet(PROCESSED / "dataset_modelado.parquet")
    data = data.sort_values(["store_nbr", "family", "week_start"]).reset_index(drop=True)
    n_antes = len(data)

    # ---------------------------------------------- 1. objetivo transformado
    data["log_ventas"] = np.log1p(data["ventas"])

    # ---------------------------------------------- 2. autorregresivas por serie
    g = data.groupby(["store_nbr", "family"], sort=False)["ventas"]

    print("[1/5] Calculando lags...")
    for L in LAGS:
        data[f"lag_{L}"] = g.shift(L)

    print("[2/5] Calculando medias moviles y desviacion (excluyendo la semana actual)...")
    ventas_pasada = g.shift(1)  # nunca incluye la semana que se predice
    for W in VENTANAS_MEDIA:
        data[f"media_movil_{W}"] = (
            ventas_pasada.groupby([data["store_nbr"], data["family"]]).rolling(W, min_periods=W).mean().values
        )
    data[f"std_{VENTANA_STD}"] = (
        ventas_pasada.groupby([data["store_nbr"], data["family"]]).rolling(VENTANA_STD, min_periods=VENTANA_STD).std().values
    )

    # ---------------------------------------------- 3. codificacion para embeddings
    print("[3/5] Codificando categoricas (family_id, store_id)...")
    familias = sorted(data["family"].unique())
    data["family_id"] = data["family"].map({f: i for i, f in enumerate(familias)}).astype("int16")
    tiendas = sorted(data["store_nbr"].unique())
    data["store_id"] = data["store_nbr"].map({s: i for i, s in enumerate(tiendas)}).astype("int16")
    print(f"       {len(familias)} familias -> family_id 0..{len(familias)-1}")
    print(f"       {len(tiendas)} tiendas   -> store_id 0..{len(tiendas)-1}")

    # ---------------------------------------------- 4. verificacion de fuga (leakage check)
    print("\n[4/5] Verificacion anti-fuga: lag_1 de la semana W debe = ventas de la semana W-1")
    muestra = data.dropna(subset=["lag_1"]).sample(min(500, len(data.dropna(subset=["lag_1"]))), random_state=42)
    fallos = 0
    for _, fila in muestra.iterrows():
        real = data[(data.store_nbr == fila.store_nbr) & (data.family == fila.family) &
                     (data.week_start == fila.week_start - pd.Timedelta(weeks=1))]
        if len(real) == 1 and not np.isclose(real["ventas"].iloc[0], fila["lag_1"]):
            fallos += 1
    assert fallos == 0, f"FUGA DETECTADA: {fallos} discrepancias en la muestra de verificacion"
    print(f"       OK — {len(muestra)} filas verificadas al azar, 0 discrepancias.")

    # ---------------------------------------------- 5. NaNs por falta de historial (inicio de serie)
    print("\n[5/5] Filas con NaN en variables autorregresivas (inicio de cada serie, sin historial suficiente):")
    cols_autoreg = [f"lag_{L}" for L in LAGS] + [f"media_movil_{W}" for W in VENTANAS_MEDIA] + [f"std_{VENTANA_STD}"]
    con_nan = data[cols_autoreg].isna().any(axis=1)
    print(f"       Total: {con_nan.sum():,} de {len(data):,} filas ({con_nan.mean()*100:.2f}%)")
    print("       Por split:")
    print(data.loc[con_nan, "split"].value_counts().to_string())
    print("\n       Por tienda, SOLO si afecta a val o test (caso de atencion: tiendas con poco historial):")
    afecta_val_test = data.loc[con_nan & data["split"].isin(["val", "test"])]
    if len(afecta_val_test):
        print(afecta_val_test.groupby("store_nbr").size().to_string())
    else:
        print("       (ninguna — todas las tiendas llegan a val/test con historial suficiente)")

    # Se eliminan las filas sin historial suficiente (no se puede evaluar ni entrenar sin lags)
    data_final = data.loc[~con_nan].reset_index(drop=True)

    out = PROCESSED / "dataset_features.parquet"
    data_final.to_parquet(out, index=False)
    print(f"\nGuardado: {out}")
    print(f"Filas: {n_antes:,} -> {len(data_final):,} ({len(data_final)/n_antes*100:.1f}% conservado)")
    print(data_final["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
