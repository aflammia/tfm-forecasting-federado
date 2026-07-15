# STATE.md — Estado actual y próxima tarea

> Documento vivo. **Actualízalo al completar cada tarea** (marca hecho, anota lo aprendido).
> Última actualización: 2026-07-15 — **T1.1, T1.2 y T1.3 cerradas.** Silos, baselines T2.2b y
> granularidad confirmados (Sesión 9); notebook 01 (Sesión 10); dataset de modelado (Sesión 11);
> variable `type` documentada con honestidad (Sesión 12); split walk-forward verificado sin fuga,
> semanas parciales excluidas (Sesión 13). Próxima: T1.4 (feature engineering — lags, medias móviles).

## ✅ Hecho

- [x] Repo creado en `Escritorio/tfm-forecasting-federado`, git inicializado (rama `main`), primer commit hecho y **subido a GitHub**: `github.com/aflammia/tfm-forecasting-federado` (privado).
- [x] `.gitignore` (protege token, datos, venv, salidas), `README.md`, `requirements.txt`.
- [x] Entorno virtual `.venv` con: kaggle, pandas, numpy, pyarrow, matplotlib, seaborn, scipy, scikit-learn, y herramientas de notebook (nbformat, nbclient, nbconvert, ipykernel).
- [x] Kernel de Jupyter registrado: `tfmfl` ("Python (tfm-forecasting-federado)").
- [x] Token de Kaggle configurado en `C:\Users\alefl\.kaggle\access_token` (autentica OK).
- [x] Dataset descargado y descomprimido en `data/raw/` (train.csv 3M filas, stores, transactions, oil, holidays).
- [x] `src/01_explore.py` — EDA inicial. 54 tiendas, 16 provincias, 33 familias, 2013→2017, 1.07B unidades. Top: Grocery I 32%, Beverages 20%.
- [x] `src/02_silo_strategy.py` — comparación de estrategias de partición. **Decisión: silos por FORMATO de tienda** (más heterogeneidad de comportamiento; escala 2.2×, promo spread 0.62).
- [x] Figura `reports/figures/01_silos_overview.png`.
- [x] **Diseño detallado del modelo y del algoritmo fijado** (Sesión 5 del cuaderno): variable objetivo (log1p de ventas semanales por tienda×familia), 9 grupos de features (sin precio — no existe en el dataset), arquitectura MLP+embeddings (~28→64→32→1), pseudocódigo y fórmula exacta de FedAvg (agregación a nivel de silo, no de tienda suelta).
- [x] `notebooks/00_refresher_regresion_lineal.ipynb` — notebook de repaso (regresión simple y múltiple desde cero, verificado contra scikit-learn, coincidencia exacta). Material de apoyo, no resultado de investigación.
- [x] Docs de traspaso: `CLAUDE.md`, `docs/PLAN.md`, este `STATE.md`, `docs/RESEARCH_LOG.md` (cuaderno de laboratorio, 7 sesiones documentadas).
- [x] Reenfoque técnico+negocio acordado con el tutor (RQ1-RQ5, baselines convencionales T2.2b, capítulos FL-vs-clean-rooms y valor de negocio en Fase 4b). Ver RESEARCH_LOG Sesión 7.
- [x] `src/03_silo_assignment_gate.py` — GATE de T1.1: counts exactos por tipo de tienda y dos propuestas de agrupamiento comparadas (no escribe fichero, solo propone).
- [x] Repo local sincronizado con la rama de la sesión móvil (`claude/thesis-project-review-isn1t8` fusionada a `main`, sin conflictos).
- [x] Criterio de silo **validado con segunda métrica** (transacciones/tráfico, correlación 0,91 con venta) — Propuesta 2 confirmada de forma robusta. Ver RESEARCH_LOG Sesión 8.
- [x] Baselines T2.2b **confirmados con fuentes 2026**: LightGBM + ETS/Holt-Winters (no ARIMA/Prophet — quedan por detrás en la evidencia). Pregunta de la Sesión 7 resuelta.
- [x] `docs/DATA.md` — diccionario de datos completo y verificado (schema, granularidad, plan de uso por fase, limitaciones). Nuevos gotchas encontrados: 8 tiendas de apertura tardía (una con solo ~4 meses de historia), huecos en transactions.csv, festivos transferidos/duplicados en holidays_events.csv.
- [x] **T1.1 cerrada**: `data/processed/stores_silos.csv` generado (Propuesta 2, confirmada por el autor). Ver `src/04_generate_silos.py`.
- [x] Granularidad familia×semana (vs. SKU×día) justificada y documentada con 4 argumentos + limitación reconocida (RESEARCH_LOG Sesión 9) — reutilizable directamente en la memoria.
- [x] `notebooks/01_refresher_mlp_embeddings_fedavg.ipynb` — MLP desde cero (94,7% menos error que regresión en datos no lineales), embeddings visualizados, simulación de FedAvg con 3 silos sintéticos (federado empata con el mejor local y bate ampliamente a los peores). Material de apoyo, ejecutado sin errores.
- [x] Entorno ampliado: `torch`, `lightgbm`, `statsmodels` instalados y en `requirements.txt`.
- [x] **T1.2 cerrada**: `data/processed/tienda_familia_semana.parquet` — agregación semanal + petróleo + festivos (nacional/regional/local, con `transferred` tratado) + día de pago + recorte de apertura tardía. 0 nulos, venta total verificada idéntica al original. Ver `src/05_build_modeling_dataset.py`.

## ▶️ PRÓXIMA TAREA — T1.4 (Fase 1)

**T1.1, T1.2 y T1.3 cerradas (2026-07-15).**
- `data/processed/stores_silos.csv` — silos Propuesta 2.
- `data/processed/tienda_familia_semana.parquet` — agregación semanal (399.762 filas, 0 nulos).
- `data/processed/dataset_modelado.parquet` — con columna `split` (train/val/test), semanas
  parciales excluidas, sin fuga temporal verificada. `configs/split_config.json` con las fechas exactas.

**T1.4 — Feature engineering**: lags (1/2/4/8 semanas), medias móviles, y preparar las variables
categóricas (familia, tienda) para los embeddings. Ver `docs/PLAN.md` y la arquitectura ya
definida en `RESEARCH_LOG.md` Sesión 5/10. Ojo: los lags deben calcularse por serie
(tienda×familia) en orden temporal, y respetar el `split` para no filtrar información de
val/test hacia atrás en el tiempo de forma incorrecta (aunque los lags en sí son pasado→futuro,
hay que verificar con un test que ningún lag "mire" hacia adelante).

## ⏳ Pendiente de decisión / acción del usuario

- **Regenerar el token de Kaggle** (se pegó en un chat; higiene). Al hacerlo, actualizar `C:\Users\alefl\.kaggle\access_token`.
- **Cuenta de Azure for Students** (verificar con correo UNAV) — necesaria para la Fase 4 (Azure ML). No bloquea las Fases 1-3.

## 🧭 Cómo ejecutar (recordatorio)

```bash
cd "/c/Users/alefl/OneDrive/Escritorio/tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/<script>.py
```
