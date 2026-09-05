"""
23_valor_negocio.py — RQ3: cuantificación del valor de negocio de colaborar sin ceder datos.

Compara la Condición A (cada tienda entrena sola) con la Condición E (federado + personalización),
que es la pregunta de negocio real del TFM: qué gana un operador por unirse a la federación en vez
de trabajar en solitario. No compara contra LightGBM, que exige juntar los datos crudos y por tanto
no está disponible en el escenario que motiva el proyecto (esa comparación se discute aparte, en el
capítulo de discusión).

PRINCIPIO DE HONESTIDAD DEL MODELO
El dataset es público (Corporación Favorita) y NO contiene precios ni márgenes: las ventas están en
unidades. Por tanto:
  1. La parte FÍSICA del cálculo (cuántas unidades de error se evitan) se deriva íntegramente de
     nuestros propios resultados, sin ningún supuesto externo.
  2. La parte MONETARIA usa parámetros de sector documentados y se presenta SIEMPRE como un rango
     con análisis de sensibilidad, nunca como una cifra única.
Esta separación es deliberada: permite que el lector acepte la primera parte aunque discuta la
segunda.

AGREGACIÓN
El error se agrega ponderado por volumen (suma de unidades de error / suma de unidades vendidas),
no por la mediana de las series: el dinero sigue al volumen, y una serie que vende 10.000 unidades
pesa más en la cuenta de resultados que una que vende 10. Por eso la mejora económica relevante no
tiene por qué coincidir con el 6,57% de mejora en la mediana del WMAPE (Sesión 31).

Entradas: reports/por_serie_condicion_A_test.csv, reports/por_serie_condicion_E_test.csv,
          data/processed/dataset_features.parquet
Salidas:  reports/valor_negocio.csv (escenarios), reports/valor_negocio_resumen.json

Ejecutar: PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/23_valor_negocio.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEMANAS_TEST = 8       # horizonte de evaluación (T1.3)
SEMANAS_ANIO = 52

# --- Parámetros de sector (con fuente; se recorren en el análisis de sensibilidad) -------------
# Margen bruto del comercio de alimentación: FMI y las cuentas de Kroger sitúan el margen bruto
# en el entorno del 21-22% de la venta. Se recorre 18-26%.
MARGEN_BRUTO = [0.18, 0.22, 0.26]
# Merma de perecederos: los informes del sector europeo sitúan entre el 5% y el 7% de los
# perecederos, que a su vez son ~50% de la venta de un supermercado (IEEP, 2024).
# `frac_error_evitable` = qué parte del error de previsión se traduce realmente en coste
# (el resto lo absorbe el stock de seguridad, la reposición intradía o el propio surtido).
FRAC_ERROR_EVITABLE = [0.20, 0.35, 0.50]
# Valor medio de la unidad vendida (€). El dataset no tiene precios: es el parámetro más
# incierto y por eso se recorre en un rango amplio y se reporta también el resultado por unidad.
VALOR_UNIDAD = [1.0, 2.0, 3.0]


def cargar_error_agregado() -> tuple[dict, pd.DataFrame]:
    """Unidades de error absoluto en test para A y E, agregadas ponderando por volumen.

    Para cada serie, WMAPE = suma|y - ŷ| / suma|y|, luego suma|y - ŷ| = WMAPE × suma|y|. Con eso
    se reconstruye el error en unidades sin necesidad de haber guardado las predicciones.

    Devuelve el agregado y el detalle por serie (necesario para el análisis de concentración)."""
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    test = features[features.split == "test"]
    ventas = (test.groupby(["store_nbr", "family"], as_index=False)["ventas"].sum()
                  .rename(columns={"ventas": "ventas_test"}))

    a = pd.read_csv(REPORTS / "por_serie_condicion_A_test.csv")[["store_nbr", "family", "wmape"]]
    e = pd.read_csv(REPORTS / "por_serie_condicion_E_test.csv")[["store_nbr", "family", "wmape"]]
    d = (a.rename(columns={"wmape": "wmape_A"})
          .merge(e.rename(columns={"wmape": "wmape_E"}), on=["store_nbr", "family"])
          .merge(ventas, on=["store_nbr", "family"]))
    # Series sin métrica válida (ventas nulas en test) no aportan al agregado.
    d = d.dropna(subset=["wmape_A", "wmape_E"])

    d["error_A"] = d["wmape_A"] * d["ventas_test"]
    d["error_E"] = d["wmape_E"] * d["ventas_test"]

    d["ahorro_unidades"] = d["error_A"] - d["error_E"]

    err_a, err_e = d["error_A"].sum(), d["error_E"].sum()
    base = {
        "n_series": int(len(d)),
        "ventas_test_unidades": float(d["ventas_test"].sum()),
        "error_A_unidades": float(err_a),
        "error_E_unidades": float(err_e),
        "unidades_evitadas_test": float(err_a - err_e),
        "wmape_agregado_A": float(err_a / d["ventas_test"].sum()),
        "wmape_agregado_E": float(err_e / d["ventas_test"].sum()),
        "mejora_relativa_agregada": float((err_a - err_e) / err_a),
        "mediana_wmape_A": float(d["wmape_A"].median()),
        "mediana_wmape_E": float(d["wmape_E"].median()),
        "n_tiendas": int(d["store_nbr"].nunique()),
    }
    return base, d


def analisis_concentracion(d: pd.DataFrame) -> dict:
    """Comprueba si el ahorro agregado descansa en unas pocas series y cuántas empeoran.

    Es la comprobación que evita sobrevender el resultado: un ahorro muy concentrado y con muchas
    series perjudicadas no se puede presentar como una mejora uniforme del negocio."""
    orden = d.sort_values("ahorro_unidades", ascending=False)
    total = orden["ahorro_unidades"].sum()
    por_tienda = d.groupby("store_nbr")["ahorro_unidades"].sum()
    negativas = d[d["ahorro_unidades"] < 0]
    return {
        "cuota_top10_series": float(orden["ahorro_unidades"].head(10).sum() / total),
        "cuota_top50_series": float(orden["ahorro_unidades"].head(50).sum() / total),
        "n_series_que_empeoran": int(len(negativas)),
        "pct_series_que_empeoran": float(len(negativas) / len(d)),
        "unidades_perdidas_en_series_que_empeoran": float(negativas["ahorro_unidades"].sum()),
        "n_tiendas_con_ahorro_positivo": int((por_tienda > 0).sum()),
        "n_tiendas_total": int(len(por_tienda)),
    }


def escenarios(base: dict) -> pd.DataFrame:
    """Malla de sensibilidad sobre los tres parámetros monetarios inciertos."""
    unidades_anio = base["unidades_evitadas_test"] * (SEMANAS_ANIO / SEMANAS_TEST)
    filas = []
    for margen in MARGEN_BRUTO:
        for frac in FRAC_ERROR_EVITABLE:
            for valor in VALOR_UNIDAD:
                # Coste unitario del error: una unidad mal prevista acaba en merma (se pierde el
                # coste del producto) o en rotura (se pierde el margen). Sin el desglose entre
                # sobre e infra-previsión, se toma el promedio de ambos costes, que es el
                # supuesto neutral: coste_merma = valor*(1-margen), coste_rotura = valor*margen.
                coste_merma = valor * (1 - margen)
                coste_rotura = valor * margen
                coste_unitario = 0.5 * (coste_merma + coste_rotura)   # = valor/2

                ahorro_red = unidades_anio * frac * coste_unitario
                filas.append({
                    "margen_bruto": margen,
                    "frac_error_evitable": frac,
                    "valor_unidad_eur": valor,
                    "coste_unitario_error_eur": coste_unitario,
                    "unidades_evitadas_anio": unidades_anio,
                    "ahorro_anual_red_eur": ahorro_red,
                    "ahorro_anual_por_tienda_eur": ahorro_red / base["n_tiendas"],
                })
    return pd.DataFrame(filas)


def main() -> None:
    base, detalle = cargar_error_agregado()
    conc = analisis_concentracion(detalle)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(REPORTS / "valor_negocio_por_serie.csv", index=False)

    print("=" * 78)
    print("Base física del cálculo (derivada solo de nuestros resultados)")
    print("=" * 78)
    print(f"  Series analizadas:            {base['n_series']:,} ({base['n_tiendas']} tiendas)")
    print(f"  Ventas en test:               {base['ventas_test_unidades']:,.0f} unidades (8 semanas)")
    print(f"  Error absoluto Condición A:   {base['error_A_unidades']:,.0f} unidades")
    print(f"  Error absoluto Condición E:   {base['error_E_unidades']:,.0f} unidades")
    print(f"  Unidades de error evitadas:   {base['unidades_evitadas_test']:,.0f} en 8 semanas")
    print(f"  WMAPE agregado A -> E:        {base['wmape_agregado_A']:.4f} -> "
          f"{base['wmape_agregado_E']:.4f}")
    print(f"  Mejora relativa (agregada):   {base['mejora_relativa_agregada']*100:.2f}%")
    print(f"  (referencia: mejora en mediana por serie = "
          f"{(base['mediana_wmape_A']-base['mediana_wmape_E'])/base['mediana_wmape_A']*100:.2f}%)")

    print()
    print("=" * 78)
    print("Robustez: ¿de dónde sale ese ahorro?")
    print("=" * 78)
    print(f"  Las 10 series con más ahorro aportan  {conc['cuota_top10_series']*100:.1f}% del total")
    print(f"  Las 50 series con más ahorro aportan  {conc['cuota_top50_series']*100:.1f}% del total")
    print(f"  Series en las que E EMPEORA a A:      {conc['n_series_que_empeoran']:,} de "
          f"{base['n_series']:,} ({conc['pct_series_que_empeoran']*100:.1f}%)")
    print(f"    con un coste conjunto de            "
          f"{abs(conc['unidades_perdidas_en_series_que_empeoran']):,.0f} unidades")
    print(f"  Tiendas con ahorro neto positivo:     "
          f"{conc['n_tiendas_con_ahorro_positivo']} de {conc['n_tiendas_total']}")

    df = escenarios(base)
    df.to_csv(REPORTS / "valor_negocio.csv", index=False)

    print()
    print("=" * 78)
    print("Sensibilidad del ahorro anual estimado (red de 54 tiendas)")
    print("=" * 78)
    print(f"  Rango completo:   {df['ahorro_anual_red_eur'].min():,.0f} € — "
          f"{df['ahorro_anual_red_eur'].max():,.0f} € / año")
    central = df[(df.margen_bruto == 0.22) & (df.frac_error_evitable == 0.35)
                 & (df.valor_unidad_eur == 2.0)].iloc[0]
    print(f"  Escenario central: {central['ahorro_anual_red_eur']:,.0f} € / año "
          f"({central['ahorro_anual_por_tienda_eur']:,.0f} € por tienda)")

    resumen = {**base, **conc,
               "ahorro_anual_min_eur": float(df["ahorro_anual_red_eur"].min()),
               "ahorro_anual_max_eur": float(df["ahorro_anual_red_eur"].max()),
               "ahorro_anual_central_eur": float(central["ahorro_anual_red_eur"]),
               "ahorro_anual_central_por_tienda_eur": float(central["ahorro_anual_por_tienda_eur"]),
               "unidades_evitadas_anio": float(df["unidades_evitadas_anio"].iloc[0])}
    with open(REPORTS / "valor_negocio_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {REPORTS / 'valor_negocio.csv'} y valor_negocio_resumen.json")


if __name__ == "__main__":
    main()
