"""
06_temporal_split.py — T1.3: corte temporal walk-forward (train/val/test).

Parte de tienda_familia_semana.parquet (T1.2), EXCLUYE las semanas parciales de los
bordes (dias_con_dato < 7 — decisión del autor, 2026-07-15, ver RESEARCH_LOG Sesión 13),
y etiqueta cada fila con su partición: test = últimas 8 semanas completas, val = las 8
anteriores, train = el resto. El corte es por FECHA global (misma frontera para todos
los silos) — se verifica después que ningún silo quede mal representado en val/test.

Salida: data/processed/dataset_modelado.parquet (+ configs/split_config.json)
"""
import json
from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

N_SEMANAS_TEST = 8
N_SEMANAS_VAL = 8


def main() -> None:
    data = pd.read_parquet(PROCESSED / "tienda_familia_semana.parquet")
    n_total_antes = len(data)

    # ---------------------------------------------- 1. excluir semanas parciales
    parciales = data["dias_con_dato"] < 7
    print(f"[1/4] Semanas parciales excluidas: {parciales.sum():,} filas "
          f"({parciales.mean()*100:.2f}% del total)")
    data = data[~parciales].drop(columns="dias_con_dato").reset_index(drop=True)

    # ---------------------------------------------- 2. definir cortes globales por fecha
    semanas = sorted(data["week_start"].unique())
    print(f"[2/4] Semanas completas disponibles: {len(semanas)} "
          f"({pd.Timestamp(semanas[0]).date()} -> {pd.Timestamp(semanas[-1]).date()})")

    semanas_test = semanas[-N_SEMANAS_TEST:]
    semanas_val = semanas[-(N_SEMANAS_TEST + N_SEMANAS_VAL):-N_SEMANAS_TEST]
    semanas_train = semanas[:-(N_SEMANAS_TEST + N_SEMANAS_VAL)]

    def etiqueta(w):
        if w in set(semanas_test):
            return "test"
        if w in set(semanas_val):
            return "val"
        return "train"

    data["split"] = data["week_start"].map(etiqueta)

    # ---------------------------------------------- 3. verificación de fugas y cobertura
    rango = data.groupby("split")["week_start"].agg(["min", "max", "nunique"])
    print("\n[3/4] Verificación de fronteras temporales (sin solape):")
    print(rango.to_string())

    assert rango.loc["train", "max"] < rango.loc["val", "min"], "FUGA: train se solapa con val"
    assert rango.loc["val", "max"] < rango.loc["test", "min"], "FUGA: val se solapa con test"
    print("OK — train < val < test, sin solape temporal.")

    print("\nFilas por split y por silo (comprobar que ningún silo se queda corto):")
    print(pd.crosstab(data["silo"], data["split"]).to_string())

    # ---------------------------------------------- 4. guardado
    CONFIGS.mkdir(parents=True, exist_ok=True)
    config = {
        "fecha_generacion": "2026-07-15",
        "semanas_parciales_excluidas": int(parciales.sum()),
        "n_semanas_test": N_SEMANAS_TEST,
        "n_semanas_val": N_SEMANAS_VAL,
        "train": {"desde": str(pd.Timestamp(rango.loc["train", "min"]).date()),
                  "hasta": str(pd.Timestamp(rango.loc["train", "max"]).date()),
                  "n_semanas": int(rango.loc["train", "nunique"])},
        "val": {"desde": str(pd.Timestamp(rango.loc["val", "min"]).date()),
                "hasta": str(pd.Timestamp(rango.loc["val", "max"]).date()),
                "n_semanas": int(rango.loc["val", "nunique"])},
        "test": {"desde": str(pd.Timestamp(rango.loc["test", "min"]).date()),
                 "hasta": str(pd.Timestamp(rango.loc["test", "max"]).date()),
                 "n_semanas": int(rango.loc["test", "nunique"])},
    }
    with open(CONFIGS / "split_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    out = PROCESSED / "dataset_modelado.parquet"
    data.to_parquet(out, index=False)

    print(f"\n[4/4] Guardado: {out}  |  configs/split_config.json")
    print(f"Filas: {n_total_antes:,} -> {len(data):,} tras excluir parciales "
          f"({len(data)/n_total_antes*100:.1f}% conservado)")
    print(data["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
