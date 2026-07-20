"""
09_report_figures_fase1.py — Figuras nuevas para el reporte de cierre de la Fase 1.
Complementa las 5 figuras ya generadas por 07_report_figures.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_ACCENT = "#2C5D7C"
C_CRIT = "#B23A48"
C_OK = "#2E7D57"
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10.5})

data = pd.read_parquet(PROCESSED / "dataset_features.parquet")

# ============================================================ FIGURA 6
# La correccion de T1.4: ventas en crudo (muy asimetrica) vs log_ventas (simetrica)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].hist(data["ventas"].clip(upper=data["ventas"].quantile(0.99)), bins=60, color=C_CRIT, alpha=0.85)
axes[0].set_title(f"ventas (bruto) — asimetría = {data['ventas'].skew():.2f}", color=C_CRIT)
axes[0].set_xlabel("ventas semanales (recortado al percentil 99 para visualizar)")
axes[0].set_ylabel("nº de filas")

axes[1].hist(data["log_ventas"], bins=60, color=C_OK, alpha=0.85)
axes[1].set_title(f"log(1+ventas) — asimetría = {data['log_ventas'].skew():.2f}", color=C_OK)
axes[1].set_xlabel("log(1+ventas)")

fig.suptitle("Por qué se corrigieron los lags en la Sesión 17: el mismo problema que resuelve\nlog(1+ventas) en el objetivo existía sin corregir en las variables de entrada", fontsize=10.5, y=1.06)
fig.tight_layout()
fig.savefig(FIG / "r6_correccion_escala_log.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[1/1] r6_correccion_escala_log.png")
