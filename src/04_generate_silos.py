"""
04_generate_silos.py — Genera data/processed/stores_silos.csv (cierre de T1.1).

Aplica la Propuesta 2 de partición en silos, confirmada por el autor el 2026-07-15
tras el GATE de T1.1 y su validación con una segunda métrica independiente
(transacciones/tráfico) — ver docs/RESEARCH_LOG.md, Sesiones 8 y 9.

  Grande  = tipo A          (9 tiendas,  venta media/tienda ~39,2M)
  Mediano = tipos D + B     (26 tiendas, venta media/tienda ~19,1M)
  Pequeño = tipos E + C     (19 tiendas, venta media/tienda ~11,8M)
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

SILO_MAP = {
    "A": "Grande",
    "D": "Mediano", "B": "Mediano",
    "E": "Pequeno", "C": "Pequeno",
}

def main() -> None:
    stores = pd.read_csv(RAW / "stores.csv")
    stores["silo"] = stores["type"].map(SILO_MAP)

    faltan = stores["silo"].isnull().sum()
    if faltan:
        raise ValueError(f"{faltan} tiendas sin silo asignado — revisa SILO_MAP")

    out_path = PROCESSED / "stores_silos.csv"
    stores.to_csv(out_path, index=False)

    print(f"Escrito: {out_path}")
    print()
    print("Reparto final de tiendas por silo:")
    print(stores.groupby("silo")["store_nbr"].count().rename("n_tiendas").to_string())
    print()
    print("Detalle tipo -> silo:")
    print(stores.groupby(["silo", "type"])["store_nbr"].count().to_string())
    print()
    print(f"Total: {len(stores)} tiendas (debe ser 54)")

if __name__ == "__main__":
    main()
