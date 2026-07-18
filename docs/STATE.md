# STATE.md — Estado actual y próxima tarea

> Documento vivo. **Actualízalo al completar cada tarea** (marca hecho, anota lo aprendido).
> Última actualización: 2026-07-18 — **T1.1 a T1.4 cerradas.** Estado del arte de la partición en
> silos documentado con cita a PA-CFL (Sesión 15); `cluster` descartado como feature de entrenamiento
> (17 clusters/54 tiendas, 4 singletons — Sesión 15); T1.4 cerrada con verificación anti-fuga (0
> discrepancias) y una corrección importante: `val` cubre 53 tiendas, no 54 (falta la tienda 52 por
> apertura reciente — Sesión 16). Sistema de citas APA creado (`docs/REFERENCIAS.md`). Próxima: T1.4b
> (decidir normalización) y T1.5 (tests de datos).

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

## ▶️ PRÓXIMA TAREA — T1.4b y T1.5 (Fase 1, cierre)

**T1.1 a T1.4 cerradas (2026-07-18).**
- `data/processed/stores_silos.csv` — silos Propuesta 2.
- `data/processed/dataset_modelado.parquet` — con columna `split`, sin fuga temporal.
- `data/processed/dataset_features.parquet` — **el dataset final**: lags, medias móviles, `log_ventas`,
  `family_id`/`store_id` codificados. 375.309 filas. **Ojo: `val` tiene 53 tiendas (no 54) — falta la
  tienda 52, ver `RESEARCH_LOG.md` Sesión 16.**

**T1.4b — Normalización:** decidir si las features continuas (lags, medias móviles) necesitan
escalado antes de entrar al MLP (los baselines LightGBM/ETS no lo necesitan). Dado que ya se trabaja
en `log_ventas`, valorar si con eso basta o si además hace falta estandarizar (z-score) por serie o
globalmente — pendiente de decidir antes de la Fase 2.

**T1.5 — Tests de datos (pytest):** formalizar como suite automatizada las verificaciones ya hechas
a mano (anti-fuga de lags, sin NaN en columnas de entrada, rangos de fecha, nº de familias/tiendas
esperado) para que corran en CI (Fase 4) y no solo como scripts sueltos.

## ⏳ Pendiente de decisión / acción del usuario

- **Regenerar el token de Kaggle** (se pegó en un chat; higiene). Al hacerlo, actualizar `C:\Users\alefl\.kaggle\access_token`.
- **Cuenta de Azure for Students** (verificar con correo UNAV) — necesaria para la Fase 4 (Azure ML). No bloquea las Fases 1-3.

## 🧭 Cómo ejecutar (recordatorio)

```bash
cd "/c/Users/alefl/OneDrive/Escritorio/tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/<script>.py
```
