"""
02_split_datos.py — Fase 4, simulacro federado real en Azure (Sesión 30).

Parte `data/processed/dataset_features.parquet` en los ficheros que consumirá cada máquina del
simulacro distribuido, de forma que **cada silo reciba SOLO sus propios datos** (el corazón del
escenario de vida real: ningún silo ve las filas de otro):

  - grande.parquet / mediano.parquet / pequeno.parquet — SOLO las filas `split=='train'` de ese
    silo. Se suben únicamente a la VM de ese silo (infra/03_deploy_datos.sh).
  - val_global.parquet — todas las filas `split=='val'` (13.992). Solo va al coordinador, que la
    usa como set de referencia para la evaluación centralizada por ronda (patrón estándar de
    server-side evaluation; ver docstring de flower_app/tfm_fl/server_app.py).

Salida: infra/datos_silos/*.parquet (gitignored — son datos, mismo criterio que data/).

Ejecutar (local, antes de tocar Azure): ./.venv/Scripts/python.exe infra/02_split_datos.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SALIDA = ROOT / "infra" / "datos_silos"

# El nombre de fichero de cada silo en minúsculas y sin tilde -- es también el que
# flower_app/tfm_fl/task.py espera (f"{silo.lower()}.parquet"); "Pequeno" ya viene sin ñ del
# pipeline de silos (stores_silos.csv), así que no hay que normalizar tildes aquí.
SILOS = ["Grande", "Mediano", "Pequeno"]


def main() -> None:
    features = pd.read_parquet(PROCESSED / "dataset_features.parquet")
    SALIDA.mkdir(parents=True, exist_ok=True)

    for silo in SILOS:
        train_silo = features[(features.silo == silo) & (features.split == "train")]
        out = SALIDA / f"{silo.lower()}.parquet"
        train_silo.to_parquet(out, index=False)
        print(f"{silo}: {len(train_silo):,} filas de train -> {out.name} "
              f"({train_silo['store_nbr'].nunique()} tiendas)")

    val_global = features[features.split == "val"]
    out_val = SALIDA / "val_global.parquet"
    val_global.to_parquet(out_val, index=False)
    print(f"val_global: {len(val_global):,} filas de val -> {out_val.name} "
          f"({val_global['store_nbr'].nunique()} tiendas)")

    # Verificación de aislamiento: la suma de filas de train de los 3 silos debe ser exactamente
    # el train total (sin solape, sin pérdida) -- si esto falla, algún silo comparte o pierde datos.
    total_train = (features.split == "train").sum()
    suma_silos = sum(len(features[(features.silo == s) & (features.split == "train")]) for s in SILOS)
    assert suma_silos == total_train, f"Partición inconsistente: {suma_silos} != {total_train}"
    print(f"\nOK — aislamiento verificado: {suma_silos:,} filas de train repartidas sin solape "
          f"entre los {len(SILOS)} silos.")


if __name__ == "__main__":
    main()
