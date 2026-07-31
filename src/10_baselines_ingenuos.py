"""
10_baselines_ingenuos.py — T2.2: baselines ingenuos (suelo de cordura).

Tres predictores sin ningún entrenamiento (solo "leer" valores ya conocidos), que cualquier
condición seria (A-F) debe superar con holgura:
  - Persistencia: predicción(semana W) = venta real de la semana W-1.
  - Estacional (52 semanas): predicción(semana W) = venta real de la misma semana, un año antes.
  - Media móvil (4 semanas): predicción(semana W) = media de las 4 semanas anteriores.

Se evalúan sobre val y test con el módulo de métricas de T2.1, por serie y agregado por silo.
Resultado guardado en reports/resultados_baselines.csv -- referencia para comparar contra A-F.

Sesión 19: las predicciones se calculan sobre un calendario semanal RECONSTRUIDO por serie (ver
calendario_semanal.py), no directamente por posición sobre las filas del dataset. Motivo: el
25-dic no tiene ninguna fila en train.csv (tiendas cerradas), lo que deja un hueco interno en
casi todas las series (1.749/1.782) tras la exclusión de "semanas parciales" en T1.3. Como
groupby().shift(N) avanza por posición y no por fecha, sin reindexar ese hueco desplazaría
silenciosamente "t-1"/"t-52"/la media móvil hacia la semana equivocada justo después de cada
Navidad. Ver Sesión 19 del RESEARCH_LOG para el diagnóstico completo.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendario_semanal import construir_calendario_completo
from metrics import metricas_por_serie, resumen

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def construir_predicciones(data: pd.DataFrame) -> pd.DataFrame:
    """Añade las 3 columnas de predicción, calculadas por serie (tienda×familia), en ESCALA
    NATURAL de ventas (no log) -- son baselines de lectura directa, no hace falta log-espacio.
    Se calculan sobre un calendario semanal reconstruido (huecos internos = NaN) para que
    shift()/rolling() -- que avanzan por posición -- no crucen un hueco silenciosamente."""
    completo = construir_calendario_completo(data, cols_valor=["ventas"])
    g = completo.groupby(["store_nbr", "family"], sort=False)["ventas"]

    completo["pred_persistencia"] = g.shift(1)
    completo["pred_estacional_52"] = g.shift(52)
    completo["pred_media_movil_4"] = g.shift(1).groupby([completo["store_nbr"], completo["family"]]).rolling(4, min_periods=4).mean().values

    cols_pred = ["pred_persistencia", "pred_estacional_52", "pred_media_movil_4"]
    data = data.merge(
        completo[["store_nbr", "family", "week_start"] + cols_pred],
        on=["store_nbr", "family", "week_start"], how="left",
    )
    return data


def evaluar_baseline(data: pd.DataFrame, col_pred: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Evalúa un baseline sobre un split (val o test), usando train de cada serie como
    referencia de escala para MASE/RMSSE."""
    df_train = data[data.split == "train"][["store_nbr", "family", "ventas"]]
    df_eval = data[(data.split == split) & data[col_pred].notna()][["store_nbr", "family", "ventas", col_pred]]
    df_eval = df_eval.rename(columns={col_pred: "prediccion"})

    por_serie = metricas_por_serie(df_eval, df_train)
    cobertura = len(df_eval) / len(data[data.split == split])
    resumen_metricas = resumen(por_serie)
    resumen_metricas["cobertura"] = cobertura
    return por_serie, resumen_metricas


def main() -> None:
    print("Cargando dataset y construyendo predicciones de los 3 baselines...")
    data = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    data = construir_predicciones(data)

    baselines = {
        "Persistencia (t-1)": "pred_persistencia",
        "Estacional (t-52)": "pred_estacional_52",
        "Media móvil (4 sem.)": "pred_media_movil_4",
    }

    filas_resumen = []
    resultados_detalle = {}

    for nombre, col in baselines.items():
        print(f"\n{'=' * 60}\n{nombre}\n{'=' * 60}")
        for split in ["val", "test"]:
            por_serie, resumen_split = evaluar_baseline(data, col, split)
            resultados_detalle[(nombre, split)] = por_serie
            print(f"  [{split}] cobertura={resumen_split['cobertura']*100:.1f}%  "
                  f"WMAPE_media={resumen_split['wmape_media']:.4f}  "
                  f"MASE_media={resumen_split['mase_media']:.4f}  "
                  f"RMSSE_media={resumen_split['rmsse_media']:.4f}")

            fila = {"baseline": nombre, "split": split}
            fila.update(resumen_split.to_dict())
            filas_resumen.append(fila)

    df_resumen = pd.DataFrame(filas_resumen)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "resultados_baselines.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\nResumen guardado en: {out}")
    print("\n" + df_resumen.to_string(index=False))

    # WMAPE agregado (no promedio de promedios) -- coherente con la definicion de WMAPE
    print("\n" + "=" * 60)
    print("WMAPE agregado global (suma de errores / suma de ventas) -- test")
    print("=" * 60)
    for nombre, col in baselines.items():
        d = data[(data.split == "test") & data[col].notna()]
        wmape_global = (d["ventas"] - d[col]).abs().sum() / d["ventas"].abs().sum()
        print(f"  {nombre}: {wmape_global:.4f}")


if __name__ == "__main__":
    main()
