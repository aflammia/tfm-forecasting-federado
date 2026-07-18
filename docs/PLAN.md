# PLAN.md — Plan detallado del proyecto

Diseñado por Opus. Ejecutado por Sonnet siguiendo el orden de tareas. Cada tarea es
autocontenida; si algo bloquea, anótalo en `STATE.md` y sigue con lo desbloqueable.

---

## 0. Objetivo y preguntas de investigación

> **Reenfoque (2026-07-14), acordado con el tutor:** el TFM debe equilibrar **técnico + negocio**.
> No basta con demostrar que el federado funciona mejor que local/centralizado — hay que demostrar
> que es más efectivo que los **métodos convencionales de forecasting en retail**, y argumentar por
> qué es preferible como solución de negocio frente a alternativas como los **data clean rooms**.
> Detalle completo de la decisión en `RESEARCH_LOG.md`, Sesión 7.

**Objetivo:** demostrar, con métricas y significancia estadística, que el aprendizaje federado
predice mejor que los métodos alternativos (locales, centralizados, y convencionales de retail) —
y argumentar, con análisis y cifras, por qué aporta un valor de negocio real frente a la
alternativa más establecida (data clean rooms).

**Preguntas de investigación (2 técnicas + 2 de negocio):**

- **RQ1** (técnica) — ¿Un modelo federado supera al local por tienda y se acerca al centralizado, sin compartir datos crudos, bajo heterogeneidad entre silos? (incluye personalización, antes RQ2)
- **RQ2** (técnica) — ¿Supera el federado a los métodos convencionales de forecasting en retail (naive estacional, suavizado exponencial ETS, y LightGBM por tienda)?
- **RQ3** (negocio) — ¿Qué valor de negocio genera la mejora de precisión del federado — reducción estimada de roturas de stock y mermas, traducida a euros?
- **RQ4** (negocio) — ¿Por qué es preferible un modelo federado a un data clean room para el forecasting colaborativo entre operadores que no comparten datos?
- **RQ5** (stretch, antes RQ3) — ¿Qué coste en precisión tiene añadir privacidad diferencial (ε)?

**Métrica estrella (técnica):** *% de la brecha Local→Centralizado que recupera el Federado*.
**Métrica estrella (negocio):** *€ ahorrados por reducción de mermas/roturas, extrapolado a la red*.

---

## 1. Diseño experimental (el corazón)

Matriz de condiciones a comparar (misma tarea, mismos datos, misma validación):

| # | Condición | Qué es | Prioridad |
|---|-----------|--------|-----------|
| A | Local por tienda | Cada tienda entrena solo con lo suyo | CORE |
| B | Centralizado por silo | Se juntan las tiendas de un mismo formato | CORE |
| C | Centralizado global | Se juntan las 3 silos (el "techo" inviable) | CORE |
| D | Federado (FedAvg) | Las silos entrenan juntas sin compartir datos | CORE |
| E | Federado + personalización | Modelo global + fine-tuning local | CORE |
| F | Federado + privacidad (DP) | Ruido diferencial, varios ε | STRETCH |

- **Métricas:** WMAPE (principal), RMSSE, MASE. Por serie (tienda×familia), por silo y agregada.
- **Validación:** walk-forward temporal. Sugerido: test = últimas ~8 semanas, val = 8 previas, resto train.
- **Significancia:** Wilcoxon pareado sobre WMAPE por serie entre condiciones (A vs D, C vs D, D vs E).
- **Modelo base:** MLP con embeddings de entidad (familia + tienda/cluster), ~28→64→32→1, Huber loss sobre log(1+ventas), Adam. Detalle completo en `RESEARCH_LOG.md`, Sesión 5. LightGBM como referencia clásica no-federada (no se federa).
- **FL jerárquico:** los **3 silos son los clientes/participantes de FedAvg** (cada silo centraliza internamente sus propias tiendas — es el mismo operador simulado). La personalización por tienda individual (condición E) es un ajuste fino *posterior* a la convergencia del federado, no un nivel adicional de FedAvg.

---

## 2. Backlog de tareas (en orden)

### FASE 1 — Pipeline de datos  ← EMPEZAR AQUÍ
- **T1.1** ✅ **Cerrada (2026-07-15).** Silos por formato: Grande=tipo A (9) · Mediano=tipos D+B (26) ·
  Pequeño=tipos E+C (19) — Propuesta 2, validada por venta y por tráfico de clientes (correlación 0,91).
  `data/processed/stores_silos.csv` generado (`src/04_generate_silos.py`). Justificación completa
  en `RESEARCH_LOG.md` Sesión 9.
- **T1.2** ✅ **Cerrada (2026-07-15).** `data/processed/tienda_familia_semana.parquet` — 399.762 filas,
  0 nulos, venta total verificada idéntica al original. Script: `src/05_build_modeling_dataset.py`.
  Detalle completo (metodología, verificaciones, decisión pendiente sobre semanas parciales) en
  `RESEARCH_LOG.md` Sesión 11.
- **T1.3** ✅ **Cerrada (2026-07-15).** Walk-forward: train 220 semanas (2013-01-07→2017-04-17) ·
  val 8 semanas (2017-04-24→2017-06-12) · test 8 semanas (2017-06-19→2017-08-07). Semanas parciales
  excluidas (2,55%). Verificado: sin fuga temporal, cobertura completa de las 54 tiendas en val/test.
  `data/processed/dataset_modelado.parquet` + `configs/split_config.json`. Detalle en `RESEARCH_LOG.md` Sesión 13.
- **T1.4** ✅ **Cerrada (2026-07-18).** Lags (1/2/4/8 sem.), medias móviles (4/8 sem.) y desviación (4 sem.)
  por serie, sin fuga (verificado sobre 500 filas). `log_ventas` y codificación `family_id`/`store_id`
  para embeddings. **Corrección importante:** `val` cubre 53 tiendas, no 54 (falta la tienda 52, apertura
  demasiado reciente para tener historial de lags en ese periodo — sí presente en `test`). Detalle en
  `RESEARCH_LOG.md` Sesión 16. `data/processed/dataset_features.parquet`.
  *(Pendiente aún: normalización por serie — decidir si hace falta dado que ya se trabaja en log-escala, ver T1.4b.)*
- **T1.4b** Decidir si hace falta normalización adicional por serie más allá de `log_ventas`, antes de
  pasar a la Fase 2 (los baselines LightGBM/ETS no la necesitan; el MLP puede beneficiarse de escalar
  las features continuas — lags, medias móviles — antes de entrar a la red).
- **T1.5** Tests de datos (pytest): sin fugas temporales, sin NaN en features de entrada, rangos de fecha correctos, familias/tiendas esperadas.

### FASE 2 — Baselines (A, B, C) + métodos convencionales de retail (RQ2)
- **T2.1** Módulo de métricas: WMAPE, RMSSE, MASE (por serie y agregada) + helper de Wilcoxon.
- **T2.2** Baselines ingenuos (naive estacional, media móvil) como suelo de cordura.
- **T2.2b** Baselines convencionales de retail — **lo que responde RQ2**: suavizado exponencial (ETS/Holt-Winters) y **LightGBM por tienda** (el estándar de facto en la industria y en competiciones de forecasting como M5). Sin esto, "supera a los métodos convencionales" quedaría sin demostrar.
- **T2.3** Condición A (Local): un modelo pequeño por tienda.
- **T2.4** Condición B (Centralizado por silo).
- **T2.5** Condición C (Centralizado global). Referencia LightGBM en paralelo.
- Integrar **Weights & Biases** desde aquí (trackear cada corrida y condición).

### FASE 3 — Federado (D, E) — resultado central
- **T3.1** Montar Flower (simulación): **cada silo = cliente** (3 participantes; cada silo centraliza internamente sus tiendas). Definir modelo (MLP+embeddings, Sesión 5), rondas, agregación ponderada por nº de filas.
- **T3.2** Condición D: FedAvg entre los 3 silos. Comparar con FedProx (robusto a no-IID).
- **T3.3** Condición E: personalización post-federado — fine-tuning de la última capa/embedding por tienda individual, estilo FedPer, sobre el modelo global ya convergido.
- **T3.4** Manejo de stragglers y muestreo de clientes.
- **T3.5** Comparación estadística A/B/C/D/E + métrica "brecha recuperada". Figuras a `reports/figures/`.

### FASE 4 — MLOps de producción
- **Config:** Hydra. **Tracking:** W&B. **Versionado datos:** DVC.
- **CI/CD:** GitHub Actions (ruff + mypy + pytest en cada push). **Contenedor:** Docker + devcontainer.
- **Registro de modelos:** Azure ML (keyword de CV). **Dashboard:** Streamlit desplegado (elige silo → ve previsión y comparación).

### FASE 4b — Capítulos de negocio (RQ3, RQ4) — análisis y escritura, sin entrenar nada
- **T4b.1** — **FL vs. Data Clean Rooms.** Comparación estructurada (tarea que resuelven, movimiento de
  datos, necesidad de tercero de confianza, aplicabilidad entre competidores directos, madurez del
  tooling, postura GDPR). Tesis a defender: los clean rooms analizan datos existentes; el FL construye
  modelos predictivos entre partes que no ceden datos — es el primitivo correcto para esta tarea, no
  un sustituto universal (reconocer la convergencia FL↔clean rooms sin restarle fuerza al argumento).
- **T4b.2** — **Cuantificación del valor de negocio.** Modelo simple: mejora de WMAPE → reducción
  estimada de roturas de stock y mermas → € por tienda y extrapolado a la red. Usar los propios
  resultados de la Fase 3 como input.

### FASE 5 — Stretch (recortar primero si falta tiempo)
- Condición F (RQ5): privacidad diferencial con **Opacus**, curva ε vs WMAPE.
- Monitorización de **drift** con Evidently. Segundo dataset (M5) como test de robustez.

### FASE 6 — Redacción y difusión
- Capítulos del TFM: intro, **estado del arte (tres bloques: métodos de forecasting + federated learning + data clean rooms)**, metodología, resultados técnicos (RQ1-RQ2), análisis de negocio (RQ3-RQ4), discusión, conclusiones.
- README impecable con diagrama y resultados. Borrador de post de LinkedIn + demo/GIF.

---

## 3. Orden de recorte si falta tiempo (<3 meses)

Nunca se toca: pipeline de datos, baselines A/B/C **+ T2.2b (ETS/LightGBM, responde RQ2)**, federado
D/E, métricas, W&B, CI, dashboard, **Fase 4b (RQ3/RQ4, análisis de negocio — es solo escritura, no
entrenar nada, así que no hay razón real para recortarla)**.
Se recorta de fuera hacia dentro: **primero** DP (F/RQ5) → **luego** segundo dataset M5 → **luego** drift/MkDocs.

**Hito clave (mitad del proyecto):** tener ya "Federado > Local y ≈ Centralizado" con significancia.
Si eso está, la tesis es defendible aunque se recorte lo demás.

---

## 4. Entregables finales

- Repo público en GitHub (README con diagrama y resultados, CI en verde, reproducible).
- Proyecto público en W&B (dashboard de todas las condiciones).
- Demo en Streamlit.
- Memoria del TFM (~50 págs).
- Post de LinkedIn + vídeo corto.
