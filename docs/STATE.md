# STATE.md — Estado actual y próxima tarea

> Documento vivo. **Actualízalo al completar cada tarea** (marca hecho, anota lo aprendido).
> Última actualización: 2026-07-20 — **FASE 1 completa y corregida** (T1.1-T1.5 + corrección Sesión
> 19: hueco interno de Navidad en 1.749/1.782 series desalineaba los lags de T1.4 por `shift()`
> posicional; corregido con `src/calendario_semanal.py`, dataset regenerado a 322.245 filas). **T2.1
> y T2.2 cerradas** (módulo de métricas + baselines ingenuos, ver Sesiones 18 y 20). Próxima: **T2.2b**
> (LightGBM + ETS/Holt-Winters, responde RQ2).

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
- [x] **T1.4b y T1.5 cerradas (2026-07-18)**: normalización (cíclica + z-score train-only) y suite de 16 tests pytest. Ver RESEARCH_LOG Sesión 17.
- [x] **Corrección Sesión 19 (2026-07-20)**: el 25-dic no tiene ninguna fila en `train.csv` para ninguna tienda/familia (cierre por festivo, sin siquiera un registro de venta=0), lo que dejaba `dias_con_dato=6` esa semana → T1.3 la excluye correctamente como "parcial" → hueco interno real en 1.749/1.782 series. `groupby().shift()` avanza por posición, no por fecha, así que ese hueco desalineaba silenciosamente los lags de T1.4 justo después de cada Navidad. Corregido con `src/calendario_semanal.py` (reindexa cada serie a un calendario semanal completo, huecos=NaN, antes de cualquier shift/rolling). `dataset_features.parquet` regenerado: **322.245 filas** (antes 375.309). Test nuevo (`test_ninguna_fila_queda_justo_despues_de_un_hueco_interno`) formaliza el invariante. 17/17 tests pasan.
- [x] **T2.1 cerrada (2026-07-18)**: `src/metrics.py` — WMAPE, MASE, RMSSE, comparación Wilcoxon pareada, `porcentaje_brecha_recuperada` (la métrica estrella del proyecto). 12/12 tests pasan. Ver RESEARCH_LOG Sesión 18.
- [x] **T2.2 cerrada (2026-07-20)**: `src/10_baselines_ingenuos.py` — persistencia (t-1), estacional (t-52), media móvil (4 sem.), evaluados en val/test. Media móvil es el suelo de cordura más fuerte (WMAPE test=0,24); estacional el más débil (WMAPE test=0,38). Resultado en `reports/resultados_baselines.csv`. Ver RESEARCH_LOG Sesión 20.

## ▶️ PRÓXIMA TAREA — T2.2b (Fase 2, RQ2)

**T1.1 a T1.5 (+ corrección Sesión 19), T2.1 y T2.2 cerradas.**
- `data/processed/stores_silos.csv` — silos Propuesta 2.
- `data/processed/dataset_modelado.parquet` — con columna `split`, sin fuga temporal.
- `data/processed/dataset_features.parquet` — **el dataset final**: lags, medias móviles, `log_ventas`,
  `family_id`/`store_id` codificados. **322.245 filas** (Sesión 19). **Ojo: `val` tiene 53 tiendas (no
  54) — falta la tienda 52, ver `RESEARCH_LOG.md` Sesión 16.**
- `reports/resultados_baselines.csv` — suelo de cordura (persistencia/estacional/media móvil).

**T2.2b — Baselines convencionales de retail (responde RQ2):** suavizado exponencial (ETS/Holt-Winters,
`statsmodels`) y LightGBM por tienda — el estándar de facto en competiciones de forecasting (M5).
Necesario para poder afirmar "supera a los métodos convencionales" con evidencia.

## ⏳ Pendiente de decisión / acción del usuario

- **Regenerar el token de Kaggle** (se pegó en un chat; higiene). Al hacerlo, actualizar `C:\Users\alefl\.kaggle\access_token`.
- **Cuenta de Azure for Students** (verificar con correo UNAV) — necesaria para la Fase 4 (Azure ML). No bloquea las Fases 1-3.

## 🧭 Cómo ejecutar (recordatorio)

```bash
cd "/c/Users/alefl/OneDrive/Escritorio/tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/<script>.py
```
