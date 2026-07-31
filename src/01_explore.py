"""
01_explore.py — Análisis exploratorio inicial de Corporación Favorita.

Objetivo: entender la estructura de tiendas y ventas, y confirmar que un
reparto en silos por región produce una heterogeneidad (no-IID) real entre
silos — condición necesaria para que el aprendizaje federado tenga sentido.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def sep(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# ---------------------------------------------------------------- tiendas
stores = pd.read_csv(RAW / "stores.csv")
sep("TIENDAS (stores.csv)")
print(f"Nº de tiendas: {len(stores)}")
print(f"Estados (provincias): {stores['state'].nunique()} | Ciudades: {stores['city'].nunique()}")
print(f"Tipos: {sorted(stores['type'].unique())} | Clusters: {stores['cluster'].nunique()}")
print("\nTiendas por estado:")
print(stores["state"].value_counts().to_string())

# ---------------------------------------------------------------- ventas
sep("VENTAS (train.csv) — cargando 3M filas...")
train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
print(f"Filas: {len(train):,}")
print(f"Rango de fechas: {train['date'].min().date()} -> {train['date'].max().date()}")
print(f"Familias de producto: {train['family'].nunique()}")
print(f"Ventas totales: {train['sales'].sum():,.0f} unidades")

fam_share = (train.groupby("family")["sales"].sum().sort_values(ascending=False))
print("\nTop 8 familias por ventas:")
print((fam_share.head(8) / fam_share.sum() * 100).round(1).astype(str).add(" %").to_string())

# ---------------------------------------------- reparto en silos por region
sep("PROPUESTA DE SILOS (por región)")
# Ecuador: agrupamos estados en 3 silos equilibrando nº de tiendas.
# Silo por peso de tiendas: los dos grandes polos (Quito=Pichincha, Guayaquil=Guayas)
# concentran la mayoría; el resto se agrupa como "otras regiones".
counts = stores["state"].value_counts()
silo_map = {}
for state in counts.index:
    if state == "Pichincha":
        silo_map[state] = "Silo_A_Sierra_Quito"
    elif state == "Guayas":
        silo_map[state] = "Silo_B_Costa_Guayaquil"
    else:
        silo_map[state] = "Silo_C_Otras_Regiones"
stores["silo"] = stores["state"].map(silo_map)
print("Tiendas por silo:")
print(stores["silo"].value_counts().to_string())

# --------------------------------------- heterogeneidad (no-IID) entre silos
sep("HETEROGENEIDAD ENTRE SILOS (¿tiene sentido el federado?)")
# Mezcla de ventas por familia en cada silo -> si difieren, hay no-IID.
t2 = train.merge(stores[["store_nbr", "silo"]], on="store_nbr")
fam_sales = t2.groupby(["silo", "family"])["sales"].sum().unstack(fill_value=0)
mix = fam_sales.div(fam_sales.sum(axis=1), axis=0)  # mezcla (%) por silo

def js_divergence(p, q):
    p, q = np.asarray(p), np.asarray(q)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

silos = mix.index.tolist()
print("Divergencia Jensen-Shannon entre la mezcla de ventas de cada par de silos")
print("(0 = idénticos, 1 = totalmente distintos):\n")
for i in range(len(silos)):
    for j in range(i + 1, len(silos)):
        d = js_divergence(mix.loc[silos[i]], mix.loc[silos[j]])
        print(f"  {silos[i]:24s} vs {silos[j]:24s}: {d:.4f}")

# nivel medio de ventas por tienda y silo (heterogeneidad de escala)
scale = (t2.groupby(["silo", "store_nbr"])["sales"].sum()
           .groupby(level=0).mean())
print("\nVenta media acumulada por tienda en cada silo (heterogeneidad de escala):")
print(scale.round(0).astype(int).to_string())

# ---------------------------------------------------------------- figura
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
stores["silo"].value_counts().plot.bar(ax=axes[0], color=["#2C5D7C", "#6B7F52", "#A66412"])
axes[0].set_title("Nº de tiendas por silo"); axes[0].set_ylabel("tiendas")
axes[0].tick_params(axis="x", rotation=20)
top_fams = fam_share.head(6).index
mix[top_fams].T.plot.bar(ax=axes[1])
axes[1].set_title("Mezcla de ventas (top familias) por silo")
axes[1].set_ylabel("% de ventas del silo"); axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "01_silos_overview.png", dpi=120)
print(f"\nFigura guardada en: {FIG / '01_silos_overview.png'}")
print("\nOK — exploración completada.")
