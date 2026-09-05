"""
24_figuras_memoria.py — genera las figuras de resultados de la memoria.

Todas las figuras salen de los CSV ya producidos por los scripts de `src/` (nada se recalcula
aquí): así cualquier cifra de una figura es trazable al experimento que la generó. Se guardan en
`memoria/figuras/` a 300 ppp y con un tamaño pensado para ocupar ~0,85 del ancho de página A4.

Criterios de diseño (constantes en todas las figuras):
  - La FORMA la elige el trabajo que hace el dato: magnitud comparada -> barras ordenadas;
    dispersión -> diagrama de caja; evolución -> líneas; parte acumulada -> curva acumulada;
    malla de parámetros -> mapa de calor secuencial.
  - El COLOR codifica una función, no decora: gris neutro para el conjunto, azul para resaltar las
    condiciones federadas (el objeto del estudio) y naranja para el mejor método no federado.
    En las figuras de una sola serie no hay leyenda; la identidad va en los ejes o en etiquetas
    directas, nunca solo en el color.
  - Sin título dentro de la figura: el pie de figura de LaTeX cumple esa función.
  - Rejilla discreta, marcas finas y sin adornos.

Ejecutar: PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/24_figuras_memoria.py
"""
# ruff: noqa: I001  (el orden de imports es deliberado: matplotlib.use("Agg") tiene que
# ejecutarse antes de importar pyplot para fijar el backend sin pantalla)
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURAS = ROOT / "memoria" / "figuras"

# Paleta categórica de referencia (slots 1-4). Los slots 1-3 están validados para todos los pares
# y los 1-4 para pares adyacentes, que es el caso de los gráficos de líneas de esta memoria.
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS = "#8a8a85"          # conjunto neutro (series sin protagonismo)
GRIS_TEXTO = "#52514e"
TINTA = "#0b0b0b"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": GRIS_TEXTO, "axes.labelcolor": TINTA,
    "xtick.color": GRIS_TEXTO, "ytick.color": GRIS_TEXTO, "text.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dcdcd8", "grid.linewidth": 0.6,
})


def _coma(x: float, dec: int = 4) -> str:
    """Formato español: separador decimal coma."""
    return f"{x:.{dec}f}".replace(".", ",")


def _eje_coma(ax, eje: str = "x", dec: int = 2) -> None:
    """Formatea las marcas del eje con coma decimal (convención española)."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:.{dec}f}".replace(".", ","))
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(fmt)


def _guardar(fig, nombre: str) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / f"{nombre}.png"
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  {ruta.name}")


# --------------------------------------------------------------------------------------------
def fig_comparativa_metodos() -> None:
    """Magnitud comparada entre métodos -> barras horizontales ordenadas.

    Es la figura de cabecera del capítulo de resultados: sitúa de un vistazo los 13 métodos del
    proyecto. El color separa tres papeles: condiciones federadas (objeto del estudio), mejor
    método no federado y el resto."""
    fuentes = [("resultados_baselines.csv", "baseline"),
               ("resultados_baselines_convencionales.csv", "baseline"),
               ("resultados_condicion_A_local.csv", "condicion"),
               ("resultados_condicion_B_silo.csv", "condicion"),
               ("resultados_condicion_C_global.csv", "condicion"),
               ("resultados_condicion_E_original_sesion27.csv", "condicion")]
    partes = []
    for fichero, col in fuentes:
        d = pd.read_csv(REPORTS / fichero).rename(columns={col: "metodo"})
        partes.append(d[d.split == "test"][["metodo", "wmape_mediana"]])
    df = pd.concat(partes, ignore_index=True)

    # La Condición D titular (FedAvg original) no tiene CSV de resumen propio con la arquitectura
    # de fábrica: se toma la mediana de la muestra COMPLETA de su fichero por serie, para que sea
    # comparable con el resto de barras (la mediana del contraste es la del subconjunto pareado).
    d_med = pd.read_csv(REPORTS / "por_serie_condicion_D_test.csv")["wmape"].median()
    df = pd.concat([df, pd.DataFrame([{"metodo": "D - Federado (FedAvg)",
                                       "wmape_mediana": d_med}])], ignore_index=True)

    df["metodo"] = df["metodo"].str.replace(r"^E - completo.*", "E - Federado + personalización",
                                            regex=True)
    df = df.sort_values("wmape_mediana", ascending=False).reset_index(drop=True)

    def color(m: str) -> str:
        if m.startswith(("D -", "E -")):
            return AZUL
        if "LightGBM" in m:
            return NARANJA
        return GRIS

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    barras = ax.barh(df["metodo"], df["wmape_mediana"],
                     color=[color(m) for m in df["metodo"]], height=0.62)
    for b, v in zip(barras, df["wmape_mediana"]):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, _coma(v),
                va="center", fontsize=7.5, color=GRIS_TEXTO)
    ax.set_xlabel("WMAPE en test (mediana por serie) — menor es mejor")
    ax.set_xlim(0, df["wmape_mediana"].max() * 1.18)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    _eje_coma(ax, "x")
    # Identidad por color explicada en texto, no solo por color.
    ax.text(1.0, -0.20, "azul: condiciones federadas   ·   naranja: mejor método no federado",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=GRIS_TEXTO)
    _guardar(fig, "resultados_comparativa_metodos")


def fig_distribucion_por_serie() -> None:
    """Dispersión entre series -> diagrama de caja.

    Las medianas por sí solas esconden que la mejora no es uniforme; esta figura muestra el rango
    intercuartílico de cada método sobre las mismas series."""
    metodos = [("condicion_A", "A · Local"), ("condicion_B", "B · Centr. silo"),
               ("condicion_C", "C · Centr. global"), ("condicion_D", "D · Federado"),
               ("condicion_E", "E · Fed. + person."), ("media_movil", "Media móvil"),
               ("lightgbm", "LightGBM")]
    datos, etiquetas = [], []
    for slug, etiqueta in metodos:
        d = pd.read_csv(REPORTS / f"por_serie_{slug}_test.csv")
        datos.append(d["wmape"].dropna().values)
        etiquetas.append(etiqueta)

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    bp = ax.boxplot(datos, tick_labels=etiquetas, showfliers=False, patch_artist=True,
                    widths=0.55, medianprops=dict(color=TINTA, linewidth=1.4))
    for parche, (slug, _) in zip(bp["boxes"], metodos):
        parche.set_facecolor(AZUL if slug in ("condicion_D", "condicion_E")
                             else NARANJA if slug == "lightgbm" else GRIS)
        parche.set_alpha(0.85); parche.set_edgecolor(GRIS_TEXTO); parche.set_linewidth(0.7)
    for elemento in ("whiskers", "caps"):
        for artista in bp[elemento]:
            artista.set_color(GRIS_TEXTO); artista.set_linewidth(0.7)
    ax.set_ylabel("WMAPE por serie (test)")
    ax.set_ylim(0, 0.45)
    _eje_coma(ax, "y")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.tick_params(axis="x", length=0)
    _guardar(fig, "resultados_distribucion_por_serie")


def fig_convergencia_federada() -> None:
    """Evolucion a lo largo de las rondas -> lineas.

    Se usa leyenda en vez de etiquetas al final de cada linea porque dos de las cuatro
    configuraciones convergen practicamente al mismo valor y las etiquetas se solapaban."""
    h = pd.read_csv(REPORTS / "historial_rondas_condicion_D.csv")
    nombres = {"fedavg_el1": "FedAvg · 1 época local", "fedavg_el2": "FedAvg · 2 épocas",
               "fedavg_el3": "FedAvg · 3 épocas", "fedprox_el2": "FedProx · 2 épocas"}
    colores = {"fedavg_el1": AZUL, "fedavg_el2": AQUA,
               "fedavg_el3": AMARILLO, "fedprox_el2": NARANJA}

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for cfg in ["fedavg_el1", "fedavg_el2", "fedavg_el3", "fedprox_el2"]:
        d = h[h.config == cfg].sort_values("ronda")
        # La ronda 0 es la inicializacion aleatoria (val_loss ~5) y aplastaria la escala.
        d = d[d.ronda >= 1]
        ax.plot(d["ronda"], d["val_loss"], color=colores[cfg], linewidth=1.6,
                label=nombres[cfg], marker="o", markersize=3)
    ax.set_xlabel("Ronda de entrenamiento federado")
    ax.set_ylabel("Pérdida de validación (Huber, val global)")
    _eje_coma(ax, "y", 2)
    ax.set_xlim(0.6, 15.4)
    ax.set_xticks(range(1, 16, 2))
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.5)
    _guardar(fig, "resultados_convergencia_federada")


def fig_significancia() -> None:
    """Tamano de efecto y significacion de los ocho contrastes -> barras divergentes.

    Polaridad (mejora/empeora) -> dos colores con el cero como referencia neutra. Las etiquetas se
    fijan a mano en lugar de derivarlas del CSV: el CSV guarda los nombres sin tildes."""
    d = pd.read_csv(REPORTS / "significancia_wilcoxon.csv")
    d["delta"] = (d["mediana_b"] - d["mediana_a"]) / d["mediana_a"] * 100

    etiquetas = {
        ("condicion_A", "condicion_D"): "A · Local  vs  D · Federado",
        ("condicion_A", "condicion_E"): "A · Local  vs  E · Fed. + person.",
        ("condicion_C", "condicion_D"): "C · Centr. global  vs  D · Federado",
        ("condicion_D", "condicion_E"): "D · Federado  vs  E · Fed. + person.",
        ("condicion_A", "condicion_B"): "A · Local  vs  B · Centr. silo",
        ("condicion_A", "condicion_C"): "A · Local  vs  C · Centr. global",
        ("condicion_E", "media_movil"): "E · Fed. + person.  vs  Media móvil",
        ("condicion_E", "lightgbm"): "E · Fed. + person.  vs  LightGBM",
    }
    d["etiqueta"] = [etiquetas[(a, b)] for a, b in zip(d["metodo_a"], d["metodo_b"])]
    d = d.sort_values("delta")

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    colores = [AZUL if v < 0 else NARANJA for v in d["delta"]]
    barras = ax.barh(d["etiqueta"], d["delta"], color=colores, height=0.6)
    for b, (_, fila) in zip(barras, d.iterrows()):
        x = fila["delta"]
        texto = "p=" + f"{fila['p_valor_holm']:.1e}".replace("e-0", "e-")
        ax.text(x + (0.7 if x >= 0 else -0.7), b.get_y() + b.get_height() / 2, texto,
                va="center", ha="left" if x >= 0 else "right",
                fontsize=7, color=GRIS_TEXTO)
    ax.axvline(0, color=GRIS_TEXTO, linewidth=0.9)
    ax.set_xlabel("Variación de la mediana del WMAPE (%) del segundo método frente al primero")
    ax.set_xlim(-21, 25)          # margen holgado para que ninguna etiqueta invada los rótulos
    _eje_coma(ax, "x", 0)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.text(1.0, -0.26, "azul: el segundo método mejora   ·   naranja: empeora",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=GRIS_TEXTO)
    _guardar(fig, "resultados_significancia")


def fig_tuning_arquitectura() -> None:
    """Resultado de la búsqueda de arquitectura -> barras ordenadas, con el punto de partida
    resaltado para que se vea cuánto se mejora sobre él."""
    t = pd.read_csv(REPORTS / "tuning_arquitectura_mlp.csv").sort_values("wmape_val")
    etiquetas = [("Original" if e == "ORIGINAL" else e.replace("candidato ", "C"))
                 for e in t["etiqueta"]]
    colores = [NARANJA if e == "ORIGINAL" else (AZUL if i == 0 else GRIS)
               for i, e in enumerate(t["etiqueta"])]

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.bar(etiquetas, t["wmape_val"], color=colores, width=0.65)
    ax.set_ylabel("WMAPE en validación (proxy agrupado)")
    _eje_coma(ax, "y", 2)
    ax.set_xlabel("Candidatos de la búsqueda, ordenados")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.text(0.99, 0.93, "naranja: arquitectura original   ·   azul: ganadora",
            transform=ax.transAxes, ha="right", fontsize=7, color=GRIS_TEXTO)
    plt.setp(ax.get_xticklabels(), fontsize=7)
    _guardar(fig, "resultados_tuning_arquitectura")


def fig_concentracion_ahorro() -> None:
    """Parte acumulada del total -> curva acumulada. Es la figura que impide sobrevender el
    resultado económico: enseña que el ahorro se concentra en pocas series."""
    d = pd.read_csv(REPORTS / "valor_negocio_por_serie.csv").sort_values(
        "ahorro_unidades", ascending=False).reset_index(drop=True)
    acum = d["ahorro_unidades"].cumsum() / d["ahorro_unidades"].sum() * 100
    x = np.arange(1, len(d) + 1)

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ax.plot(x, acum, color=AZUL, linewidth=1.8)
    ax.axhline(100, color=GRIS_TEXTO, linewidth=0.8, linestyle=":")
    for n, desplaza in ((50, -22), (100, 10)):
        ax.plot([n, n], [0, acum.iloc[n - 1]], color=GRIS, linewidth=0.8, linestyle="--")
        ax.annotate(f"{n} series: {acum.iloc[n-1]:.0f}%".replace(".", ","),
                    (n, acum.iloc[n - 1]), xytext=(12, desplaza), textcoords="offset points",
                    fontsize=7.5, color=GRIS_TEXTO)
    ax.set_xlabel("Series ordenadas por ahorro aportado (de mayor a menor)")
    ax.set_ylabel("Porcentaje acumulado del ahorro total (%)")
    ax.set_xlim(0, len(d)); ax.set_ylim(0, 130)
    ax.grid(True); ax.set_axisbelow(True)
    _guardar(fig, "negocio_concentracion_ahorro")


def fig_sensibilidad_valor() -> None:
    """Malla de dos parámetros -> mapa de calor secuencial (un solo tono, claro a oscuro)."""
    d = pd.read_csv(REPORTS / "valor_negocio.csv")
    d = d[d.margen_bruto == 0.22]
    tabla = d.pivot_table(index="frac_error_evitable", columns="valor_unidad_eur",
                          values="ahorro_anual_red_eur")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    im = ax.imshow(tabla.values / 1e6, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tabla.columns)),
                  [f"{c:.0f} €".replace(".", ",") for c in tabla.columns])
    ax.set_yticks(range(len(tabla.index)), [f"{i:.0%}" for i in tabla.index])
    ax.set_xlabel("Valor medio de la unidad vendida")
    ax.set_ylabel("Fracción del error\nque genera coste")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.values[i, j] / 1e6
            ax.text(j, i, _coma(v, 2), ha="center", va="center", fontsize=8,
                    color="white" if v > tabla.values.max() / 1e6 * 0.6 else TINTA)
    cb = fig.colorbar(im, ax=ax, label="Ahorro anual estimado (M€)")
    _eje_coma(cb.ax, "y", 1)
    ax.set_xticks(np.arange(-0.5, len(tabla.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tabla.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    _guardar(fig, "negocio_sensibilidad")


def main() -> None:
    print("Generando figuras en memoria/figuras/ ...")
    fig_comparativa_metodos()
    fig_distribucion_por_serie()
    fig_convergencia_federada()
    fig_significancia()
    fig_tuning_arquitectura()
    fig_concentracion_ahorro()
    fig_sensibilidad_valor()
    print("Listo.")


if __name__ == "__main__":
    main()
