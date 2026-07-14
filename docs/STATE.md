# STATE.md — Estado actual y próxima tarea

> Documento vivo. **Actualízalo al completar cada tarea** (marca hecho, anota lo aprendido).
> Última actualización: 2026-07-14 — reenfoque técnico+negocio acordado con el tutor (ver RESEARCH_LOG Sesión 7); GATE de T1.1 ejecutado (counts por tipo ya mostrados), pendiente de que el autor elija entre Propuesta 1 y Propuesta 2 de agrupamiento.

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

## ▶️ PRÓXIMA TAREA — T1.1 (Fase 1)

**Asignación final de silos por formato.** GATE ejecutado (2026-07-14) con `src/03_silo_assignment_gate.py`.
Counts reales por tipo:

| Tipo | Tiendas | Venta media/tienda |
|---|---|---|
| A | 9 | 39.227.094 |
| D | 18 | 19.504.628 |
| B | 8 | 18.157.579 |
| E | 4 | 14.955.609 |
| C | 15 | 10.962.316 |

Dos propuestas sobre la mesa (**pendiente de que el autor elija una, o proponga otra**):
- **Propuesta 1** (2/1/2, original): Grande=A+D (27 t.) · Mediano=B (8 t.) · Pequeño=E+C (19 t.)
- **Propuesta 2** (1/2/2, recomendada — respeta el salto natural A→D, mayor que D→B): Grande=A (9 t.) · Mediano=D+B (26 t.) · Pequeño=E+C (19 t.)

Una vez el autor elija:
1. Generar `data/processed/stores_silos.csv` con `store_nbr, city, state, type, cluster, silo`.
2. Imprimir el reparto final y confirmar que ningún silo queda con muy pocas tiendas.

Después seguir con T1.2 (dataset tienda×familia×semana) — ver `docs/PLAN.md`.

## ⏳ Pendiente de decisión / acción del usuario

- **Elegir Propuesta 1 vs. 2 de agrupamiento de silos** (arriba) — bloquea T1.1 y toda la Fase 1.
- **¿Incluir ARIMA/Prophet además de ETS/LightGBM en los baselines de T2.2b?** (pregunta abierta de la Sesión 7 del RESEARCH_LOG, reenfoque técnico+negocio).
- **Regenerar el token de Kaggle** (se pegó en un chat; higiene). Al hacerlo, actualizar `C:\Users\alefl\.kaggle\access_token`.
- **Cuenta de Azure for Students** (verificar con correo UNAV) — necesaria para la Fase 4 (Azure ML). No bloquea las Fases 1-3.

## 🧭 Cómo ejecutar (recordatorio)

```bash
cd "/c/Users/alefl/OneDrive/Escritorio/tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/<script>.py
```
