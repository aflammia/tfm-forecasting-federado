"""
00_download_data.py — Descarga reproducible y portable del dataset Favorita.

Funciona igual en local (Windows, token en archivo) y en la nube (Linux, token en
la variable de entorno KAGGLE_API_TOKEN). Idempotente: si los datos ya están, no hace nada.

Uso:
    python src/00_download_data.py
"""
import zipfile
from pathlib import Path

COMPETITION = "store-sales-time-series-forecasting"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = ["train.csv", "stores.csv", "transactions.csv", "oil.csv", "holidays_events.csv"]


def datos_presentes() -> bool:
    return all((RAW / f).exists() for f in EXPECTED)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if datos_presentes():
        print("Datos ya presentes en data/raw/ — nada que descargar.")
        return

    # La API de Kaggle lee automáticamente el token de KAGGLE_API_TOKEN
    # o del archivo ~/.kaggle/access_token (o kaggle.json).
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Descargando '{COMPETITION}' en {RAW} ...")
    api.competition_download_files(COMPETITION, path=str(RAW), quiet=False)

    zip_path = RAW / f"{COMPETITION}.zip"
    print("Descomprimiendo ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)

    if datos_presentes():
        print("OK — datos descargados y descomprimidos.")
    else:
        raise RuntimeError("La descarga terminó pero faltan archivos esperados.")


if __name__ == "__main__":
    main()
