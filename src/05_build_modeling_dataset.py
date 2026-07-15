"""
05_build_modeling_dataset.py — T1.2: construye el dataset de modelado.

Agrega train.csv a tienda x familia x semana, recorta el historial de tiendas de
apertura tardía, y une petróleo (semanal) y festivos (nacional/regional/local, con
tratamiento correcto de `transferred`) y una señal de "semana con día de pago".

Salida: data/processed/tienda_familia_semana.parquet
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def semana_lunes(fecha: pd.Series) -> pd.Series:
    """Devuelve el lunes de la semana ISO de cada fecha."""
    return fecha - pd.to_timedelta(fecha.dt.weekday, unit="D")


def main() -> None:
    # ---------------------------------------------------------------- carga
    train = pd.read_csv(
        RAW / "train.csv", parse_dates=["date"],
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
    )
    stores = pd.read_csv(PROCESSED / "stores_silos.csv")
    oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
    hol = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])

    # ---------------------------------------------- 1. recorte por apertura tardía
    apertura = train[train.sales > 0].groupby("store_nbr").date.min().rename("fecha_apertura")
    n_antes = len(train)
    train = train.merge(apertura, on="store_nbr")
    train = train[train.date >= train.fecha_apertura].drop(columns="fecha_apertura")
    print(f"[1/6] Recorte por apertura tardía: {n_antes - len(train):,} filas eliminadas "
          f"({(n_antes - len(train)) / n_antes * 100:.2f}% del total)")

    # ---------------------------------------------- 2. agregación a semana
    train["week_start"] = semana_lunes(train["date"])
    agg = (train.groupby(["store_nbr", "family", "week_start"], as_index=False)
                 .agg(ventas=("sales", "sum"),
                      onpromotion=("onpromotion", "mean"),
                      dias_con_dato=("sales", "count")))
    print(f"[2/6] Agregado a semana: {len(agg):,} filas "
          f"({agg.store_nbr.nunique()} tiendas x {agg.family.nunique()} familias x "
          f"{agg.week_start.nunique()} semanas)")

    # ---------------------------------------------- 3. metadatos de tienda/silo
    data = agg.merge(stores[["store_nbr", "silo", "city", "state", "type"]], on="store_nbr", how="left")
    assert data["silo"].isnull().sum() == 0, "Hay tiendas sin silo asignado"
    print(f"[3/6] Unido con stores_silos.csv — silos: {data.silo.value_counts().to_dict()}")

    # ---------------------------------------------- 4. petróleo semanal (con relleno de huecos)
    cal = pd.DataFrame({"date": pd.date_range(oil.date.min(), oil.date.max(), freq="D")})
    oil_full = cal.merge(oil, on="date", how="left")
    oil_full["dcoilwtico"] = oil_full["dcoilwtico"].ffill().bfill()
    oil_full["week_start"] = semana_lunes(oil_full["date"])
    oil_weekly = (oil_full.groupby("week_start", as_index=False)["dcoilwtico"]
                          .mean().rename(columns={"dcoilwtico": "oil_price"}))
    data = data.merge(oil_weekly, on="week_start", how="left")
    print(f"[4/6] Petróleo unido — nulos en oil_price: {data['oil_price'].isnull().sum()}")

    # ---------------------------------------------- 5. festivos (nacional/regional/local)
    # Un festivo "transferred=True" NO se observa esa fecha (el día real está en el
    # registro type=Transfer correspondiente) -> se excluye. 'Work Day' es una
    # recuperación de puente, no un festivo -> se excluye de esta señal.
    hol_obs = hol[(hol.type != "Work Day") & ~((hol.type == "Holiday") & (hol.transferred == True))].copy()
    hol_obs["week_start"] = semana_lunes(hol_obs["date"])

    semanas_nacional = set(hol_obs.loc[hol_obs.locale == "National", "week_start"])
    data["es_festivo_nacional"] = data["week_start"].isin(semanas_nacional)

    reg = (hol_obs.loc[hol_obs.locale == "Regional", ["locale_name", "week_start"]]
                  .drop_duplicates().rename(columns={"locale_name": "state"}))
    reg["es_festivo_regional"] = True
    data = data.merge(reg, on=["state", "week_start"], how="left")
    data["es_festivo_regional"] = data["es_festivo_regional"].fillna(False)

    loc = (hol_obs.loc[hol_obs.locale == "Local", ["locale_name", "week_start"]]
                  .drop_duplicates().rename(columns={"locale_name": "city"}))
    loc["es_festivo_local"] = True
    data = data.merge(loc, on=["city", "week_start"], how="left")
    data["es_festivo_local"] = data["es_festivo_local"].fillna(False)

    print(f"[5/6] Festivos: {data.es_festivo_nacional.sum():,} filas con festivo nacional, "
          f"{data.es_festivo_regional.sum():,} regional, {data.es_festivo_local.sum():,} local "
          f"(sobre {len(data):,} filas totales)")

    # ---------------------------------------------- 6. calendario (semana del año, mes, día de pago)
    data["semana_del_anio"] = data["week_start"].dt.isocalendar().week.astype(int)
    data["mes"] = data["week_start"].dt.month
    data["week_end"] = data["week_start"] + pd.Timedelta(days=6)

    dia15 = pd.to_datetime(dict(year=data["week_start"].dt.year, month=data["mes"], day=15))
    fin_mes = data["week_start"].dt.to_period("M").dt.to_timestamp("M")

    data["semana_con_dia_pago"] = (
        ((dia15 >= data["week_start"]) & (dia15 <= data["week_end"])) |
        ((fin_mes >= data["week_start"]) & (fin_mes <= data["week_end"]))
    )
    data = data.drop(columns="week_end")
    print(f"[6/6] Calendario añadido — {data.semana_con_dia_pago.sum():,} filas en semana con día de pago")

    # ---------------------------------------------------------------- guardado
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "tienda_familia_semana.parquet"
    data.to_parquet(out, index=False)

    print()
    print("=" * 60)
    print(f"Guardado: {out}")
    print(f"Shape final: {data.shape}")
    print(f"Rango de semanas: {data.week_start.min().date()} -> {data.week_start.max().date()}")
    print(f"Nulos por columna:\n{data.isnull().sum()[data.isnull().sum() > 0]}")
    print("=" * 60)
    print("\nMuestra:")
    print(data.sample(5, random_state=1).to_string())


if __name__ == "__main__":
    main()
