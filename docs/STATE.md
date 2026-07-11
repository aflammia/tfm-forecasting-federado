# STATE.md — Estado actual y próxima tarea

> Documento vivo. **Actualízalo al completar cada tarea** (marca hecho, anota lo aprendido).
> Última actualización: fin de la fase de exploración (por Opus).

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
- [x] Docs de traspaso: `CLAUDE.md`, `docs/PLAN.md`, este `STATE.md`, `docs/RESEARCH_LOG.md` (cuaderno de laboratorio, 6 sesiones documentadas).

## ▶️ PRÓXIMA TAREA — T1.1 (Fase 1)

**Asignación final de silos por formato.**
1. Calcular nº de tiendas por tipo (A-E) y venta media por tipo (ya se vio: orden por escala A>D>B>E>C).
2. Reagrupar los 5 tipos en 3 formatos balanceados (el grupo "Mediano" quedó con solo 8 tiendas en la
   versión provisional; buscar un reparto más equilibrado, p. ej. Grande=[A], Mediano=[D,B], Pequeño=[E,C], o similar).
3. Guardar `data/processed/stores_silos.csv` con `store_nbr, city, state, type, cluster, silo`.
4. Imprimir el reparto final y confirmar que ningún silo queda con muy pocas tiendas.

Después seguir con T1.2 (dataset tienda×familia×semana) — ver `docs/PLAN.md`.

## ⏳ Pendiente de decisión / acción del usuario

- **Confirmar silos por formato** (Opus lo recomendó; el usuario aún no dijo un "sí" explícito — validar antes de construir mucho encima).
- **Regenerar el token de Kaggle** (se pegó en un chat; higiene). Al hacerlo, actualizar `C:\Users\alefl\.kaggle\access_token`.
- **Cuenta de Azure for Students** (verificar con correo UNAV) — necesaria para la Fase 4 (Azure ML). No bloquea las Fases 1-3.
- **Primer commit de git** — recomendable hacer un checkpoint (`git add . && git commit`). Aún no hecho.

## 🧭 Cómo ejecutar (recordatorio)

```bash
cd "/c/Users/alefl/OneDrive/Escritorio/tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/<script>.py
```
