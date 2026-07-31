"""
02_silo_strategy.py — ¿Cómo definir los silos?

Compara tres formas de partir las 54 tiendas en silos y mide cuál produce
la heterogeneidad (no-IID) más rica y defendible:
  - por REGIÓN (provincia agrupada)
  - por FORMATO de tienda (type A-E de Favorita)
  - por CLUSTER (agrupación propia de Favorita)

Métricas de heterogeneidad entre silos:
  - Divergencia Jensen-Shannon media de la MEZCLA de categorías (comportamiento)
  - Ratio de ESCALA (venta media por tienda: mayor / menor silo)
  - Diferencia de INTENSIDAD PROMOCIONAL entre silos
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

stores = pd.read_csv(RAW / "stores.csv")
train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])

# escala por tipo para ordenar los formatos (A..E) de mayor a menor venta
store_tot = train.groupby("store_nbr")["sales"].sum()
type_scale = (stores.set_index("store_nbr")["type"].map(lambda x: x)
              .to_frame().assign(sales=store_tot).groupby("type")["sales"].mean()
              .sort_values(ascending=False))

def region_silo(s):
    return {"Pichincha": "R_Quito", "Guayas": "R_Guayaquil"}.get(s, "R_Otras")

# formato: agrupamos los 5 tipos en 3 "formatos" segun su escala real
order = list(type_scale.index)          # p.ej. ['A','D','B','E','C'] de mayor a menor
big, mid, small = order[:2], order[2:3], order[3:]
def format_silo(t):
    if t in big: return "F_Grande"
    if t in mid: return "F_Mediano"
    return "F_Pequeno"

# cluster: 17 clusters -> agrupamos en 3 tercios por escala media del cluster
clu_scale = (stores.set_index("store_nbr")["cluster"].to_frame()
             .assign(sales=store_tot).groupby("cluster")["sales"].mean().sort_values())
terc = np.array_split(clu_scale.index, 3)
clu_map = {c: f"C_{i+1}" for i, arr in enumerate(terc) for c in arr}

stores["by_region"] = stores["state"].map(region_silo)
stores["by_format"] = stores["type"].map(format_silo)
stores["by_cluster"] = stores["cluster"].map(clu_map)

t = train.merge(stores, on="store_nbr")

def js(p, q):
    p, q = np.asarray(p), np.asarray(q); m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def evaluate(col):
    fam = t.groupby([col, "family"])["sales"].sum().unstack(fill_value=0)
    mix = fam.div(fam.sum(axis=1), axis=0)
    silos = mix.index.tolist()
    js_avg = np.mean([js(mix.loc[a], mix.loc[b]) for a, b in combinations(silos, 2)])
    scale = t.groupby([col, "store_nbr"])["sales"].sum().groupby(level=0).mean()
    scale_ratio = scale.max() / scale.min()
    promo = t.groupby(col)["onpromotion"].mean()
    promo_spread = promo.max() - promo.min()
    counts = stores[col].value_counts().sort_index()
    return silos, js_avg, scale_ratio, promo_spread, counts

print("=" * 74)
print(f"{'ESTRATEGIA':<14}{'nº silos':<9}{'JS mezcla':<11}{'ratio escala':<14}{'promo spread'}")
print("=" * 74)
for name, col in [("Región", "by_region"), ("Formato", "by_format"), ("Cluster", "by_cluster")]:
    silos, js_avg, sr, ps, counts = evaluate(col)
    print(f"{name:<14}{len(silos):<9}{js_avg:<11.4f}{sr:<14.2f}{ps:.3f}")
    print(f"   tiendas por silo: {counts.to_dict()}")
print("=" * 74)
print("\nLectura: JS mezcla alto = silos con distinto comportamiento de compra;")
print("ratio escala alto = distinto tamaño; promo spread alto = distinta política.")
print(f"\n(Orden real de formatos por escala, mayor->menor: {order})")
