# CLAUDE.md — Orientación para sesiones de Claude Code

> **Si eres una sesión nueva (p. ej. Sonnet retomando el trabajo de Opus): lee este archivo,
> luego `docs/PLAN.md` (plan completo) y `docs/STATE.md` (estado actual + próxima tarea).**
> El plan lo diseñó Opus al detalle; tu trabajo es ejecutar las tareas de `docs/STATE.md` en orden.

## Qué es este proyecto

TFM: **previsión de demanda federada en supermercados**. Demostrar con métricas y rigor
estadístico que un modelo **federado** predice mejor que un modelo **local por tienda** y casi
tan bien como uno **centralizado** (que entre negocios independientes sería inviable). Objetivo
paralelo: un stack **MLOps de producción** que impresione a reclutadores de Big Tech (LinkedIn).

**Tesis a demostrar:** `Local < Federado ≈ Centralizado`, y la personalización mejora los silos más atípicos.

## Decisiones ya tomadas (NO reabrir sin motivo)

- **Dataset:** Corporación Favorita — Kaggle `store-sales-time-series-forecasting`. Ya descargado en `data/raw/`.
- **Granularidad de predicción:** tienda × familia de producto × **semana** (33 familias, 54 tiendas). No SKU individual (demasiado; familia es lo limpio).
- **Silos:** por **FORMATO de tienda** (tipo A-E de Favorita agrupado en 3 formatos: Grande / Mediano / Pequeño). Es el reparto con más heterogeneidad de comportamiento y el que mejor se explica ("hiper / super / proximidad" ≈ Alcampo/Mercadona/DIA). *Decisión de trabajo — confirmada por análisis en `src/02_silo_strategy.py`.*
- **Federado jerárquico:** tienda → modelo por formato (silo) → modelo global federado.
- **Cómputo:** Azure for Students (gratis, 100 $) como principal; Kaggle GPU gratis como banco de pruebas. Modelos pequeños → poca GPU.
- **Modelo federable:** red neuronal pequeña (MLP/LSTM/TCN, PyTorch) porque los pesos se promedian en FedAvg. LightGBM solo como baseline clásico de referencia (NO se federa).
- **Framework FL:** Flower (`flwr`).

## Convenciones técnicas (IMPORTANTE)

- **Python del venv:** ejecuta siempre con `./.venv/Scripts/python.exe` (Windows, Git Bash). No uses el Python del sistema.
- **Consola:** antepon `PYTHONIOENCODING=utf-8` a los scripts o los acentos salen mal.
- **Token de Kaggle:** en `C:\Users\alefl\.kaggle\access_token` (formato nuevo KGAT_). **NUNCA** lo subas al repo (ya está en `.gitignore`).
- **Datos:** `data/raw/` (crudos, gitignored) y `data/processed/` (procesados, gitignored). Los CSV/parquet no van al repo.
- **Figuras:** a `reports/figures/`. **Código:** a `src/`. **Configs:** a `configs/`.
- **Validación temporal:** SIEMPRE walk-forward por fecha. NUNCA split aleatorio (es serie temporal).

## Gotchas descubiertos (ahorra tiempo)

- La **mezcla de categorías es homogénea** entre regiones/silos en Ecuador. La heterogeneidad real (no-IID) está en la **escala** (2-3×) y en la **intensidad promocional**. Enmarca el no-IID así, no como "distinto mix de producto".
- Favorita tiene quirks conocidos a tener en cuenta al modelar: **terremoto de abril 2016** (pico de ventas), **días de pago** (15 y fin de mes suben ventas), economía dependiente del **petróleo** (columna oil), y **algunas tiendas abren tarde** (no todas tienen histórico completo desde 2013).
- Hay ventas en 0 y familias sin apenas movimiento en tiendas pequeñas → usar WMAPE (no MAPE, que explota con ceros).

## Entornos de ejecución (LOCAL vs NUBE) — léelo antes de ejecutar nada

Este proyecto se trabaja desde dos sitios. Detecta en cuál estás y usa los comandos correctos:

**A) Local — PC Windows del autor** (sesiones de escritorio):
- Python del venv: `./.venv/Scripts/python.exe`
- Token de Kaggle: archivo `C:\Users\alefl\.kaggle\access_token`
- Datos ya presentes en `data/raw/`
- Antepón `PYTHONIOENCODING=utf-8` a los scripts (consola Windows).

**B) Nube — Claude Code on the web / móvil** (VM Linux efímera):
- El repo se clona SIN datos, SIN `.venv` y SIN token (todo está gitignored). Hay que reconstruir.
- Python del sistema directamente (`python`), tras instalar deps: `pip install -r requirements.txt` (ideal: dejarlo en el *Setup script* del entorno cloud).
- Token de Kaggle: variable de entorno `KAGGLE_API_TOKEN` (configurada en el panel del entorno cloud), no archivo.
- Datos: ejecutar `python src/00_download_data.py` (idempotente; lee el token de la env var o del archivo). Requiere que el dominio de Kaggle esté permitido en el acceso de red del entorno.

**Regla práctica de reparto de trabajo:** la nube/móvil es ideal para **planificar, redactar la memoria, escribir/revisar código y revisar PRs** (no necesitan los datos). Las **ejecuciones pesadas de datos y entrenamiento** conviene hacerlas en local o en el cómputo de Azure previsto — no en la VM efímera de la nube.

## Regla de documentación (OBLIGATORIA)

Esto es un TFM: **si no está documentado, no cuenta.** Cada paso con resultados (un script que
corre, un experimento, una decisión metodológica) DEBE registrarse en **`docs/RESEARCH_LOG.md`**
con el estándar de una sección de Métodos/Resultados de artículo científico: objetivo, método
exacto, comandos reproducibles, resultados con cifras precisas, interpretación y justificación de
las decisiones. Usa la plantilla del final de ese archivo. El historial es **inmutable**: no
edites entradas cerradas; si algo se corrige, añade una entrada nueva que lo referencie.

Este cuaderno es la materia prima de los capítulos de Metodología y Resultados de la memoria.

## Regla de citación (OBLIGATORIA)

Cada vez que se investigue algo apoyándose en una fuente externa (paper, dataset, documentación
técnica, artículo) y se cite en `RESEARCH_LOG.md`, **hay que añadir también su entrada completa en
`docs/REFERENCIAS.md`, en formato APA (7.ª edición)**, en el momento — no dejarlo para el final.
Antes de dar por buena una cita, **verificar los datos exactos** (autores, año, título, venue) con
una búsqueda si no se tienen con certeza — nunca inventar ni aproximar una referencia. Este archivo
se copia directamente a la bibliografía de la memoria.

## Estado y próximos pasos

Ver **`docs/STATE.md`** — documento vivo, actualízalo al completar cada tarea.
Ver **`docs/RESEARCH_LOG.md`** — cuaderno de laboratorio, añade una entrada por cada paso con resultados.
Ver **`docs/DATA.md`** — diccionario de datos: qué contiene cada fichero, granularidad, gotchas conocidos (tiendas de apertura tardía, festivos transferidos, etc.) y el plan de uso de los datos mapeado a cada fase. Léelo antes de tocar cualquier fichero de `data/raw/`.
Ver **`docs/REFERENCIAS.md`** — bibliografía APA de todas las fuentes externas citadas, lista para la memoria.
