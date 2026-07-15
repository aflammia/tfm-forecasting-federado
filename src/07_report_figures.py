"""
07_report_figures.py — Genera las figuras del reporte de progreso, a partir de datos reales
del proyecto (no maquetas). Guarda PNG en reports/figures/.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Paleta consistente con la documentación del proyecto
C_ACCENT = "#2C5D7C"
C_AMBER = "#A66412"
C_OK = "#2E7D57"
C_CRIT = "#B23A48"
C_GRANDE, C_MEDIANO, C_PEQUENO = "#2C5D7C", "#6B7F52", "#A66412"
plt.rcParams.update({
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
    "font.size": 10.5, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
})

with open(ROOT / "configs" / "split_config.json", encoding="utf-8") as f:
    split_cfg = json.load(f)

data = pd.read_parquet(PROCESSED / "dataset_modelado.parquet")
stores = pd.read_csv(PROCESSED / "stores_silos.csv")

# ============================================================ FIGURA 1
# Ventas semanales totales, con fronteras train/val/test y el terremoto
semanal = data.groupby("week_start", as_index=False)["ventas"].sum()

fig, ax = plt.subplots(figsize=(11, 4.3))
ax.plot(semanal.week_start, semanal.ventas, color=C_ACCENT, linewidth=1.1)

train_fin = pd.Timestamp(split_cfg["train"]["hasta"])
val_ini = pd.Timestamp(split_cfg["val"]["desde"])
val_fin = pd.Timestamp(split_cfg["val"]["hasta"])
test_ini = pd.Timestamp(split_cfg["test"]["desde"])

ax.axvspan(val_ini, val_fin, color=C_AMBER, alpha=0.15, label="validación (8 sem.)")
ax.axvspan(test_ini, semanal.week_start.max(), color=C_CRIT, alpha=0.13, label="test (8 sem.)")
ax.set_ylim(top=semanal.ventas.max() * 1.22)
ax.annotate("Terremoto Ecuador\n(abril 2016)", xy=(pd.Timestamp("2016-04-18"), semanal.ventas.max() * 0.965),
            xytext=(pd.Timestamp("2014-09-01"), semanal.ventas.max() * 1.15),
            arrowprops=dict(arrowstyle="->", color=C_CRIT, lw=1.2), fontsize=9, color=C_CRIT, ha="center")

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Ventas semanales totales (unidades)")
ax.set_title("Ventas semanales agregadas — dataset completo, con el corte train/val/test", pad=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig(FIG / "r1_serie_semanal_split.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[1/5] r1_serie_semanal_split.png")

# ============================================================ FIGURA 2
# Composicion de silos: nº tiendas y venta media
train_sales = pd.read_csv(RAW / "train.csv", usecols=["store_nbr", "sales"])
venta_tienda = train_sales.groupby("store_nbr")["sales"].sum()
stores2 = stores.merge(venta_tienda.rename("venta_total"), on="store_nbr")
resumen_silo = stores2.groupby("silo").agg(n_tiendas=("store_nbr", "count"),
                                            venta_media=("venta_total", "mean")).reindex(
                                            ["Grande", "Mediano", "Pequeno"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colores = [C_GRANDE, C_MEDIANO, C_PEQUENO]
axes[0].bar(resumen_silo.index, resumen_silo.n_tiendas, color=colores)
axes[0].set_title("Nº de tiendas por silo")
for i, v in enumerate(resumen_silo.n_tiendas):
    axes[0].text(i, v + 0.4, str(int(v)), ha="center", fontsize=10)

axes[1].bar(resumen_silo.index, resumen_silo.venta_media, color=colores)
axes[1].set_title("Venta media acumulada por tienda")
axes[1].ticklabel_format(style="plain", axis="y")
for i, v in enumerate(resumen_silo.venta_media):
    axes[1].text(i, v + 5e5, f"{v/1e6:.1f}M", ha="center", fontsize=9)

fig.suptitle("Composición de los 3 silos (Propuesta 2)", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "r2_composicion_silos.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[2/5] r2_composicion_silos.png")

# ============================================================ FIGURA 3
# Venta vs transacciones por tipo (evidencia del salto natural A | D,B | E,C)
trans = pd.read_csv(RAW / "transactions.csv")
trans_avg = trans.groupby("store_nbr")["transactions"].mean()
stores3 = stores2.merge(trans_avg.rename("trans_media"), on="store_nbr")
por_tipo = stores3.groupby("type").agg(venta_media=("venta_total", "mean"),
                                        trans_media=("trans_media", "mean")).sort_values("venta_media", ascending=False)

fig, ax1 = plt.subplots(figsize=(8, 4.3))
x = np.arange(len(por_tipo))
ax1.bar(x - 0.19, por_tipo.venta_media / 1e6, width=0.38, color=C_ACCENT, label="Venta media/tienda (M)")
ax1.set_ylabel("Venta media por tienda (millones)", color=C_ACCENT)
ax1.set_xticks(x); ax1.set_xticklabels(por_tipo.index)
ax1.set_xlabel("type")

ax2 = ax1.twinx()
ax2.bar(x + 0.19, por_tipo.trans_media, width=0.38, color=C_AMBER, label="Transacciones/día")
ax2.set_ylabel("Transacciones medias/día", color=C_AMBER)
ax2.grid(False)

ax1.axvline(0.5, color=C_CRIT, linestyle="--", linewidth=1.3)

fig.suptitle("Por qué Grande = {A} y no {A, D}: el salto natural está tras el tipo A", fontsize=11, y=1.05)
fig.text(0.5, 0.965, "la línea roja marca el mayor salto en ambas métricas — no entre D y B",
          ha="center", fontsize=8.7, color=C_CRIT, style="italic")
fig.tight_layout()
fig.savefig(FIG / "r3_venta_vs_transacciones.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[3/5] r3_venta_vs_transacciones.png")

# ============================================================ FIGURA 4
# Heatmap type x cluster (anidamiento)
ct = pd.crosstab(stores.type, stores.cluster)
fig, ax = plt.subplots(figsize=(9, 3.2))
im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns)
ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
ax.set_xlabel("cluster"); ax.set_ylabel("type")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > ct.values.max()/2 else C_ACCENT, fontsize=8)
ax.set_title("cluster anida casi perfectamente dentro de type (16 de 17 clusters, un único type)")
fig.tight_layout()
fig.savefig(FIG / "r4_type_cluster_heatmap.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[4/5] r4_type_cluster_heatmap.png")

# ============================================================ FIGURA 5
# FedAvg local vs. federado (resultados verificados del notebook 01, Sesion 10)
resultados_fedavg = {"Local\n(Grande)": 122.91, "Local\n(Mediano)": 209.54,
                      "Local\n(Pequeño)": 534.75, "FEDERADO": 123.53}
fig, ax = plt.subplots(figsize=(7, 4.2))
colores_barra = [C_CRIT, C_CRIT, C_CRIT, C_OK]
barras = ax.bar(resultados_fedavg.keys(), resultados_fedavg.values(), color=colores_barra)
for b, v in zip(barras, resultados_fedavg.values()):
    ax.text(b.get_x() + b.get_width()/2, v + 8, f"{v:.1f}", ha="center", fontsize=10)
ax.set_ylabel("MSE en validación conjunta (simulación de juguete)")
ax.set_title("El federado empata con el mejor silo local y bate ampliamente a los otros dos\n(notebook 01, datos sintéticos)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "r5_fedavg_toy_resultado.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[5/5] r5_fedavg_toy_resultado.png")

print("\nTodas las figuras guardadas en:", FIG)
