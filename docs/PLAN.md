# PLAN.md — Plan detallado del proyecto

Diseñado por Opus. Ejecutado por Sonnet siguiendo el orden de tareas. Cada tarea es
autocontenida; si algo bloquea, anótalo en `STATE.md` y sigue con lo desbloqueable.

---

## 0. Objetivo y preguntas de investigación

**Objetivo:** demostrar, con métricas y significancia estadística, el valor del aprendizaje
federado para previsión de demanda en una red de supermercados independientes.

- **RQ1** — ¿Un modelo federado supera al local por tienda y se acerca al centralizado, sin compartir datos crudos?
- **RQ2** — ¿La personalización (modelo global + cabeza local) mejora los silos más atípicos?
- **RQ3** — (stretch) ¿Qué coste en precisión tiene añadir privacidad diferencial (ε)?

**Métrica estrella:** *% de la brecha Local→Centralizado que recupera el Federado* (un número que resume la tesis).

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
- **Modelo base:** red pequeña en PyTorch (empezar con MLP sobre features tabulares + lags; considerar LSTM/TCN si mejora). LightGBM como referencia clásica no-federada.
- **FL jerárquico:** clientes = tiendas; agregación dentro de silo (formato) y luego entre silos.

---

## 2. Backlog de tareas (en orden)

### FASE 1 — Pipeline de datos  ← EMPEZAR AQUÍ
- **T1.1** Asignación final de silos por formato. Reagrupar tipos A-E en 3 formatos balanceados
  (revisar counts por tipo; evitar que un silo quede con muy pocas tiendas). Guardar
  `data/processed/stores_silos.csv` con columna `silo`.
- **T1.2** Construir dataset de modelado: agregar `train.csv` a **tienda×familia×semana** (suma de `sales`,
  media de `onpromotion`). Unir covariables: precio del petróleo (semanal), flags de festivo
  (de `holidays_events.csv`, cuidado con `locale`: Nacional/Regional/Local y `transferred`), y features
  de calendario (semana del año, mes, ¿es semana con día de pago?). Guardar parquet en `data/processed/`.
- **T1.3** Corte temporal walk-forward por fecha (train/val/test). Guardar el corte en `configs/`.
- **T1.4** Feature engineering: lags (1,2,4,8 semanas), medias móviles, promo, calendario, festivos, oil.
  Documentar cada feature. Normalización por serie (para manejar la heterogeneidad de escala entre silos).
- **T1.5** Tests de datos (pytest): sin fugas temporales, sin NaN en features de entrada, rangos de fecha correctos, familias/tiendas esperadas.

### FASE 2 — Baselines (A, B, C)
- **T2.1** Módulo de métricas: WMAPE, RMSSE, MASE (por serie y agregada) + helper de Wilcoxon.
- **T2.2** Baselines ingenuos (naive estacional, media móvil) como suelo de cordura.
- **T2.3** Condición A (Local): un modelo pequeño por tienda.
- **T2.4** Condición B (Centralizado por silo).
- **T2.5** Condición C (Centralizado global). Referencia LightGBM en paralelo.
- Integrar **Weights & Biases** desde aquí (trackear cada corrida y condición).

### FASE 3 — Federado (D, E) — resultado central
- **T3.1** Montar Flower (simulación): cada tienda = cliente. Definir modelo, rondas, muestreo.
- **T3.2** Condición D: FedAvg plano y **jerárquico** (tiendas→silo→global). Comparar con FedProx (robusto a no-IID).
- **T3.3** Condición E: personalización (global + cabeza local por tienda/silo, estilo FedPer).
- **T3.4** Manejo de stragglers y muestreo de clientes.
- **T3.5** Comparación estadística A/B/C/D/E + métrica "brecha recuperada". Figuras a `reports/figures/`.

### FASE 4 — MLOps de producción
- **Config:** Hydra. **Tracking:** W&B. **Versionado datos:** DVC.
- **CI/CD:** GitHub Actions (ruff + mypy + pytest en cada push). **Contenedor:** Docker + devcontainer.
- **Registro de modelos:** Azure ML (keyword de CV). **Dashboard:** Streamlit desplegado (elige silo → ve previsión y comparación).

### FASE 5 — Stretch (recortar primero si falta tiempo)
- Condición F: privacidad diferencial con **Opacus**, curva ε vs WMAPE.
- Monitorización de **drift** con Evidently. Segundo dataset (M5) como test de robustez.

### FASE 6 — Redacción y difusión
- Capítulos del TFM (intro, estado del arte, metodología, resultados, discusión, conclusiones).
- README impecable con diagrama y resultados. Borrador de post de LinkedIn + demo/GIF.

---

## 3. Orden de recorte si falta tiempo (<3 meses)

Nunca se toca: pipeline de datos, baselines A/B/C, federado D/E, métricas, W&B, CI, dashboard.
Se recorta de fuera hacia dentro: **primero** DP (F) → **luego** segundo dataset M5 → **luego** drift/MkDocs.

**Hito clave (mitad del proyecto):** tener ya "Federado > Local y ≈ Centralizado" con significancia.
Si eso está, la tesis es defendible aunque se recorte lo demás.

---

## 4. Entregables finales

- Repo público en GitHub (README con diagrama y resultados, CI en verde, reproducible).
- Proyecto público en W&B (dashboard de todas las condiciones).
- Demo en Streamlit.
- Memoria del TFM (~50 págs).
- Post de LinkedIn + vídeo corto.
