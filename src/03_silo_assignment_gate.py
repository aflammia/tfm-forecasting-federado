"""
03_silo_assignment_gate.py — GATE de T1.1: asignación final de silos por formato.

Imprime el desglose exacto tipo -> nº tiendas -> venta media (lo que el autor pidió
ver antes de fijar el agrupamiento) y propone un reparto en 3 formatos balanceados.

Este script SOLO IMPRIME una propuesta — no escribe data/processed/stores_silos.csv.
Ese fichero se genera en un paso aparte, tras la confirmación explícita del autor.
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

stores = pd.read_csv(RAW / "stores.csv")
train = pd.read_csv(RAW / "train.csv", usecols=["store_nbr", "sales"])

store_total = train.groupby("store_nbr")["sales"].sum()
stores = stores.merge(store_total.rename("venta_total"), on="store_nbr")

por_tipo = (
    stores.groupby("type")
    .agg(n_tiendas=("store_nbr", "count"), venta_media_tienda=("venta_total", "mean"))
    .sort_values("venta_media_tienda", ascending=False)
)
por_tipo["venta_media_tienda"] = por_tipo["venta_media_tienda"].round(0).astype(int)

print("=" * 60)
print("DESGLOSE EXACTO POR TIPO DE TIENDA (ordenado por venta media, desc.)")
print("=" * 60)
print(por_tipo.to_string())
print()
print(f"Total tiendas: {por_tipo['n_tiendas'].sum()}  (debe ser 54)")
print(f"Orden de escala: {' > '.join(por_tipo.index)}")

# --- dos propuestas de agrupamiento en 3 formatos, para comparar ---
propuestas = {
    "Propuesta 1 (2/1/2 — la del script original)": {
        "Grande": ["A", "D"], "Mediano": ["B"], "Pequeño": ["E", "C"],
    },
    "Propuesta 2 (1/2/2 — más equilibrada en nº de tipos)": {
        "Grande": ["A"], "Mediano": ["D", "B"], "Pequeño": ["E", "C"],
    },
}

print()
print("=" * 60)
print("PROPUESTAS DE AGRUPAMIENTO (comparadas, ninguna se aplica todavía)")
print("=" * 60)
for nombre, grupos in propuestas.items():
    print(f"\n{nombre}:")
    for silo, tipos in grupos.items():
        sub = por_tipo.loc[tipos]
        n = sub["n_tiendas"].sum()
        # venta media ponderada por nº de tiendas de cada tipo dentro del silo
        venta_media = (sub["n_tiendas"] * sub["venta_media_tienda"]).sum() / n
        print(f"  {silo:9s} tipos={tipos}  tiendas={n:2d}  venta_media_tienda≈{venta_media:,.0f}")

print()
print("=" * 60)
print("NO se ha escrito ningún fichero. Pendiente de confirmación del autor")
print("sobre qué propuesta (u otra) usar antes de generar stores_silos.csv.")
print("=" * 60)
