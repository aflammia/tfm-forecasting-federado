FROM python:3.11-slim

WORKDIR /app

# Capa de dependencias separada del código para aprovechar la caché de Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY conf/ conf/
COPY tests/ tests/
COPY pyproject.toml .

# Los datos NO van en la imagen (data/raw y data/processed son gitignored, pesados y del autor) --
# se montan como volumen o se descargan dentro con src/00_download_data.py + KAGGLE_API_TOKEN,
# igual que el flujo "nube" ya documentado en CLAUDE.md.
ENV PYTHONIOENCODING=utf-8
ENTRYPOINT ["python"]
CMD ["-m", "pytest", "tests/", "-v"]
