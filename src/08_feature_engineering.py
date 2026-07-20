"""
08_feature_engineering.py — T1.4 + T1.4b: ingeniería de variables y normalización.

Añade sobre dataset_modelado.parquet (T1.2+T1.3):
  - Variables autorregresivas calculadas en ESCALA LOGARITMICA (lag_1..8, medias moviles,
    desviacion) -- consistente con la justificacion de log(1+ventas) de la Sesion 5: si los
    lags se dejaran en escala cruda, se reintroduciria la misma heterocedasticidad y el mismo
    problema de escala entre silos que el log-transform del objetivo resuelve.
  - onpromotion tambien en log(1+x) -- asimetria muy alta (skew=11.07) en escala cruda.
  - Codificacion ciclica (seno/coseno) de semana_del_anio y mes -- para que el modelo entienda
    que la semana 52 esta cerca de la semana 1, no lejos (un codigo ordinal simple no captura eso).
  - Estandarizacion z-score de TODAS las features continuas finales, con media/desviacion
    calculadas SOLO con el conjunto de train (para no filtrar informacion de val/test).
  - Objetivo transformado log_ventas, y codificacion family_id/store_id para embeddings.

Salida: data/processed/dataset_features.parquet + configs/normalizacion.json (medias/desv. de train,
para poder revertir o aplicar la misma transformacion de forma reproducible).

CORRECCION (Sesion 19): los lags/medias moviles se calculan sobre un calendario semanal
RECONSTRUIDO por serie (ver calendario_semanal.py), no directamente sobre las filas de
dataset_modelado.parquet. Motivo: el 25-dic no tiene NINGUNA fila en train.csv (tiendas
cerradas), lo que deja `dias_con_dato=6` en la semana que lo contiene y esa semana se excluye
en T1.3 como "parcial" -- creando un hueco interno en casi todas las series (1749/1782). Como
groupby().shift(N) avanza por POSICION y no por fecha, sin reindexar ese hueco desplazaba
silenciosamente los lags de las semanas siguientes (p.ej. lag_log_1 pasaba a ser en realidad
la venta de 2 semanas atras). Ver Sesion 19 del RESEARCH_LOG para el diagnostico completo.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendario_semanal import construir_calendario_completo

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

LAGS = [1, 2, 4, 8]
VENTANAS_MEDIA = [4, 8]
VENTANA_STD = 4


def main() -> None:
    data = pd.read_parquet(PROCESSED / "dataset_modelado.parquet")
    data = data.sort_values(["store_nbr", "family", "week_start"]).reset_index(drop=True)
    n_antes = len(data)

    # ---------------------------------------------- 1. objetivo transformado
    data["log_ventas"] = np.log1p(data["ventas"])

    # ---------------------------------------------- 2. autorregresivas EN ESCALA LOG, por serie
    # Sobre un calendario semanal RECONSTRUIDO (huecos internos, p.ej. Navidad, = NaN) para que
    # shift()/rolling() -- que avanzan por POSICION -- no crucen un hueco silenciosamente.
    print("[1/6] Reconstruyendo calendario semanal completo por serie (huecos internos = NaN)...")
    completo = construir_calendario_completo(data, cols_valor=["log_ventas"])
    n_huecos = completo["log_ventas"].isna().sum()
    print(f"       Semanas con hueco interno reintroducidas como NaN: {n_huecos:,}")

    print("[2/6] Calculando lags sobre log_ventas (no sobre ventas en bruto)...")
    g = completo.groupby(["store_nbr", "family"], sort=False)["log_ventas"]
    for L in LAGS:
        completo[f"lag_log_{L}"] = g.shift(L)

    print("       Medias moviles y desviacion (excluyendo la semana actual)...")
    log_pasado = g.shift(1)
    for W in VENTANAS_MEDIA:
        completo[f"media_movil_log_{W}"] = (
            log_pasado.groupby([completo["store_nbr"], completo["family"]]).rolling(W, min_periods=W).mean().values
        )
    completo[f"std_log_{VENTANA_STD}"] = (
        log_pasado.groupby([completo["store_nbr"], completo["family"]]).rolling(VENTANA_STD, min_periods=VENTANA_STD).std().values
    )

    cols_autoreg_calc = [f"lag_log_{L}" for L in LAGS] + [f"media_movil_log_{W}" for W in VENTANAS_MEDIA] + [f"std_log_{VENTANA_STD}"]
    data = data.merge(
        completo[["store_nbr", "family", "week_start"] + cols_autoreg_calc],
        on=["store_nbr", "family", "week_start"], how="left",
    )

    # ---------------------------------------------- 3. onpromotion en log(1+x) -- muy asimetrico en crudo
    data["log_onpromotion"] = np.log1p(data["onpromotion"])

    # ---------------------------------------------- 4. codificacion ciclica de calendario
    print("[3/7] Codificacion ciclica (seno/coseno) de semana del anio y mes...")
    data["semana_sin"] = np.sin(2 * np.pi * data["semana_del_anio"] / 52)
    data["semana_cos"] = np.cos(2 * np.pi * data["semana_del_anio"] / 52)
    data["mes_sin"] = np.sin(2 * np.pi * data["mes"] / 12)
    data["mes_cos"] = np.cos(2 * np.pi * data["mes"] / 12)

    # ---------------------------------------------- 5. categoricas para embeddings
    print("[4/7] Codificando categoricas (family_id, store_id)...")
    familias = sorted(data["family"].unique())
    data["family_id"] = data["family"].map({f: i for i, f in enumerate(familias)}).astype("int16")
    tiendas = sorted(data["store_nbr"].unique())
    data["store_id"] = data["store_nbr"].map({s: i for i, s in enumerate(tiendas)}).astype("int16")

    # ---------------------------------------------- 6. verificacion anti-fuga (sobre la nueva version log)
    print("\n[5/7] Verificacion anti-fuga: lag_log_1 de la semana W debe = log_ventas de la semana W-1")
    con_lag1 = data.dropna(subset=["lag_log_1"])
    muestra = con_lag1.sample(min(500, len(con_lag1)), random_state=42)
    fallos = 0
    for _, fila in muestra.iterrows():
        real = data[(data.store_nbr == fila.store_nbr) & (data.family == fila.family) &
                     (data.week_start == fila.week_start - pd.Timedelta(weeks=1))]
        if len(real) == 1 and not np.isclose(real["log_ventas"].iloc[0], fila["lag_log_1"]):
            fallos += 1
    assert fallos == 0, f"FUGA DETECTADA: {fallos} discrepancias en la muestra de verificacion"
    print(f"       OK — {len(muestra)} filas verificadas al azar, 0 discrepancias.")

    # ---------------------------------------------- filtrar filas sin historial suficiente
    # (incluye ahora, correctamente, las filas justo despues de un hueco interno tipo Navidad,
    # que antes de la Sesion 19 se quedaban con un lag mal alineado en vez de ir a NaN)
    cols_autoreg = cols_autoreg_calc
    con_nan = data[cols_autoreg].isna().any(axis=1)
    print(f"\n[6/7] Filas sin historial suficiente: {con_nan.sum():,} ({con_nan.mean()*100:.2f}%)")
    print(data.loc[con_nan, "split"].value_counts().to_string())
    data = data.loc[~con_nan].reset_index(drop=True)

    # ---------------------------------------------- 7. estandarizacion z-score (SOLO estadisticos de train)
    print("\n[7/7] Estandarizando features continuas (media/desviacion calculadas SOLO con train)...")
    cols_a_normalizar = (
        cols_autoreg + ["log_onpromotion", "oil_price"]
    )
    stats = {}
    train_mask = data["split"] == "train"
    for c in cols_a_normalizar:
        mu = data.loc[train_mask, c].mean()
        sigma = data.loc[train_mask, c].std()
        stats[c] = {"media": float(mu), "desviacion": float(sigma)}
        data[f"{c}_z"] = (data[c] - mu) / sigma

    CONFIGS.mkdir(parents=True, exist_ok=True)
    with open(CONFIGS / "normalizacion.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"       Estadísticos guardados en configs/normalizacion.json ({len(stats)} columnas)")

    # verificacion: media~0, std~1 en train; val/test pueden diferir levemente (es lo esperado)
    for c in cols_a_normalizar[:3]:
        m = data.loc[train_mask, f"{c}_z"].mean()
        s = data.loc[train_mask, f"{c}_z"].std()
        print(f"       {c}_z en train: media={m:.4f} (~0) std={s:.4f} (~1)")

    out = PROCESSED / "dataset_features.parquet"
    data.to_parquet(out, index=False)
    print(f"\nGuardado: {out}")
    print(f"Filas: {n_antes:,} -> {len(data):,} ({len(data)/n_antes*100:.1f}% conservado)")
    print(data["split"].value_counts().to_string())
    print(f"\nColumnas finales de features continuas listas para el modelo: {[c+'_z' for c in cols_a_normalizar] + ['semana_sin','semana_cos','mes_sin','mes_cos']}")


if __name__ == "__main__":
    main()
