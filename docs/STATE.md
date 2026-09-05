# STATE.md — Estado actual y próxima tarea

> Documento vivo. **Actualízalo al completar cada tarea** (marca hecho, anota lo aprendido).
> Última actualización: 2026-09-05 — **FASE 1, FASE 2, FASE 3 (core) y FASE 4 (MLOps) completas**,
> incluido el **simulacro federado REAL en Azure EJECUTADO** (3 VMs, dashboard en Azure ML, modelo
> registrado), más el tuning de D/E. T1.1-T1.5 (+ corrección Sesión 19); T2.1-T2.5; T3.1-T3.3 (Flower, Condición
> D FedAvg/FedProx, Condición E personalización); Sesión 28 (tuning); Sesión 29 (CI, Docker,
> dashboard Streamlit, DVC, Hydra); Sesión 30 (simulacro federado REAL en Azure con el Deployment
> Engine de Flower — 3 VMs, dashboard en Azure ML; equivalencia verificada en local, pasos de Azure
> pendientes de `az login` del autor). **Resultado central del TFM (RQ1) sigue siendo la
> configuración de la Sesión 27: E (federado + personalización) — WMAPE test mediana 0,1379
> (regenerado y confirmado en la Sesión 28) — supera a A (local), B y C (centralizado) y D
> (federado sin personalizar)**, y empata con el 2º mejor método del proyecto (media móvil,
> 0,1379), por detrás solo de LightGBM (0,132). El tuning de la Sesión 28 SÍ mejoró D por sí solo
> (0,1507→0,1402, ~7%) pero esa mejora no se traslada a E — todas las variantes de personalización
> con la arquitectura tuneada quedaron peor que el original (sobreajuste probable en los datasets
> pequeños del fine-tuning por tienda); la corrección de sesgo de Duan por tienda resultó
> catastrófica (0,1446→0,8876, atípicos con pocas filas) y se descartó. Hallazgos negativos
> honestos, documentados en Sesión 28 — no se fuerza una mejora que no se dio. La métrica de
> "brecha recuperada" (T2.1) da un % negativo mecánicamente porque su premisa (centralizado =
> techo) no se cumple en este dataset heterogéneo — la lectura correcta es que E bate en absoluto
> tanto a local como a centralizado, más fuerte que "recuperar una brecha". Se encontró y corrigió
> un bug de aliasing de memoria en `get_params()` (Sesión 26). **T3.5 (contraste de Wilcoxon
> completo) se cerró en la Sesión 31** y la **Fase 4b de negocio, en la Sesión 32**, que además dejó
> redactada la memoria del TFM (`memoria/main.pdf`, 59 páginas tras eliminar el capítulo de
> Presupuesto por no ser un requisito de la normativa de MADI). Queda pendiente únicamente la
> revisión de la tutora. T3.4 (stragglers) sigue siendo una extensión opcional no abordada.

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
- [x] **T2.2b cerrada (2026-07-21)**: `src/11_baselines_convencionales.py` — ETS/Holt-Winters por serie. Diagnóstico y corrección de una inestabilidad seria en ETS (tendencia sin amortiguar + prefijos de ceros estructurales por falta de surtido de familia → predicciones de millones de unidades para PRODUCE en varias tiendas); tras `damped_trend`, recorte de prefijo de ceros y techo de cordura (3× el máximo histórico de cada serie), la cola pesada restante es la limitación conocida de ETS ante demanda intermitente (PLAYERS AND ELECTRONICS, CELEBRATION, PET SUPPLIES...), no un bug. Nuevo gotcha de datos documentado: 779/1.749 series (45%) tienen un prefijo de ceros >4 semanas al inicio de train — no todas las tiendas venden todas las familias (`DATA.md` 2.6.3). Ver RESEARCH_LOG Sesión 21.
- [x] **Corrección Sesión 22 (2026-07-21)**: el primer LightGBM (por tienda, 54 modelos, sin tuning) no superaba ni a la media móvil (T2.2). Tras tunear hiperparámetros (búsqueda aleatoria, 25 candidatos) y ampliar features (festivos, día de pago), seguía sin ganar — causa real: cada modelo por tienda entrenaba con ~5-6 mil filas y decidía early stopping sobre un val de solo ~264 filas, demasiado ruidoso. Sustituido por **un único LightGBM global** (`store_id`+`family_id` como categóricas, ~54x más datos por modelo) — bate a la media móvil y a ETS en WMAPE/MASE/RMSSE por mediana (WMAPE test 0,132 vs 0,138 y 0,334), cobertura 100%. `configs/lightgbm_hiperparametros.json` guardado para reproducibilidad. Ver RESEARCH_LOG Sesión 22.
- [x] **T2.3 cerrada (2026-07-21)**: `src/modelo_mlp.py` — MLP+embeddings en PyTorch (33→64→32→1, 4.985 parámetros, Huber+Adam, early stopping), arquitectura compartida por las condiciones A-E, reutilizable sin cambios. `src/12_condicion_a_local.py` — Condición A (Local): 54 modelos independientes, uno por tienda. WMAPE test mediana=0,1476 (por delante de persistencia/estacional/ETS, por detrás de media móvil y LightGBM global — esperado, cada modelo ve solo ~6.000 filas propias). A diferencia de la corrección del LightGBM (Sesión 22), aquí la escasez de datos por tienda NO se corrige: es justo lo que la condición "Local" debe representar. 7 tests nuevos (46/46 en total). Ver RESEARCH_LOG Sesión 23.
- [x] **T2.4-T2.5 cerradas (2026-07-21)**: `src/13_condicion_b_silo.py` (Condición B, 3 modelos, uno por silo) y `src/14_condicion_c_global.py` (Condición C, 1 modelo global). **Hallazgo no trivial:** WMAPE test mediana empeora monótonamente con más centralización — A=0,1476 < B=0,1676 < C=0,1736 — pese a que C ve ~54× más datos de entrenamiento que cualquier modelo de A. Se verificó que NO es sub-entrenamiento (más paciencia en el silo Grande solo produce sobreajuste: val_loss mínimo idéntico, luego sube). Es heterogeneidad no-IID real entre tiendas de un mismo silo — motiva directamente la personalización de la condición E (T3.3). Ver RESEARCH_LOG Sesión 24.
- [x] **Weights & Biases integrado (2026-07-22)**: `entrenar()` (`modelo_mlp.py`) acepta un `wandb_run` opcional (hook desacoplado, el módulo no importa `wandb`). `src/15_wandb_resumen_fase2.py` registra los 8 métodos de la Fase 2 ya cerrados (sin reentrenar nada) en modo **offline** — no hay cuenta de W&B configurada todavía en esta máquina. 4 tests nuevos (50/50 en total). Ver RESEARCH_LOG Sesión 25.
- [x] **T3.1-T3.2 cerradas (2026-07-22)**: `src/federado_flower.py` — Flower (`flwr[simulation]` 1.32.1, backend Ray) montado; los 3 silos son los clientes, checkpointing manual por ronda (evaluación centralizada), soporte FedAvg y FedProx. **Bug encontrado y corregido:** `get_params()` compartía memoria con los pesos en vivo del modelo (`.numpy()` sobre CPU es zero-copy) — detectado por los tests, no afectó a la simulación real (la serialización entre procesos de Ray rompe el alias como efecto secundario). Condición D: FedAvg gana a FedProx (μ=0,01 sin tunear) — WMAPE test mediana 0,1507 vs 0,1681. **D ya bate a B (0,168) y C (0,174)** sin compartir datos crudos. 5 tests nuevos. Ver RESEARCH_LOG Sesión 26.
- [x] **T3.3 cerrada (2026-07-22)**: `src/17_condicion_e_personalizacion.py` — fine-tuning por tienda desde los pesos de D (FedAvg). **WMAPE test mediana 0,1396 — supera a A, B, C y D**, la mejor de las 5 condiciones del proyecto y 3ª mejor de los 11 métodos comparados (solo por detrás de LightGBM 0,132 y media móvil 0,138). La métrica de brecha recuperada da un % negativo mecánicamente (su premisa de que Centralizado es el techo no se cumple aquí) — la lectura correcta es que E bate en absoluto a local y a centralizado. `entrenar()` ganó el parámetro `modelo_inicial` para continuar desde pesos dados. 1 test nuevo (56/56 en total). Ver RESEARCH_LOG Sesión 27. **Fase 3 (core) completa.**
- [x] **Tuning de D/E cerrado (2026-07-30)**: `ArquitecturaMLP` (dataclass) hace configurable la arquitectura del MLP en todo el pipeline (`modelo_mlp.py`, `federado_flower.py`); `src/18_tuning_arquitectura.py` (búsqueda aleatoria, 13 candidatos) encuentra una arquitectura mejor para el proxy agrupado (WMAPE val 0,1880→0,1093); `src/16_condicion_d_federado.py` confirma la mejora en D real (test 0,1507→0,1402 con FedProx epocas_locales=2); `src/17_condicion_e_personalizacion.py` prueba 3 variantes de personalización (fine-tuning completo/lr bajo/estilo FedPer con `congelar_base`) — **ninguna supera al resultado original** (0,1446 la mejor nueva vs. 0,1379 el original regenerado); `src/correccion_sesgo.py`+`src/19_correccion_duan.py` prueban la corrección de Duan (1983) — **catastrófica** (0,1446→0,8876) por sensibilidad a atípicos con pocas filas por tienda, descartada. **Se mantiene la configuración de la Sesión 27 como el mejor resultado federado del proyecto.** 26 tests nuevos (89/89 en total). Ver RESEARCH_LOG Sesión 28.
- [x] **Fase 4 (MLOps) cerrada, excepto Azure ML (2026-07-31)**: CI (`.github/workflows/ci.yml`: ruff+mypy informativo+pytest, con guard para saltar tests que dependen de datos reales); Docker (`Dockerfile`+`.devcontainer/`, build verificado, ejecución pendiente por caída del daemon local); dashboard Streamlit (`dashboard/app.py`, 3 pestañas, lee `reports/*.csv` ya generados); DVC (`dvc.yaml`, 5 etapas, remoto **local** en `../dvc-storage-tfm/` fuera del repo — funciona solo en esta máquina, repuntable a Azure Blob más adelante; `dvc repro` verificado de extremo a extremo, reproduce cifras idénticas a las de las Sesiones 11-19); Hydra (`conf/`, alcance acotado a los scripts 12/13/14/16/18 — A/B/C, tuning, D — que de verdad se benefician; 17/19 quedan sin Hydra por su acoplamiento cruzado ya existente, decisión justificada en RESEARCH_LOG). mypy encontró y corrigió 3 discrepancias reales de tipo en `modelo_mlp.py`/`federado_flower.py`. 72/72 tests pasan. Ver RESEARCH_LOG Sesión 29.
- [x] **Simulacro federado REAL en Azure — EJECUTADO de extremo a extremo (2026-08-08)**: paso del Simulation Engine (un proceso) al **Deployment Engine** de Flower — 3 VMs B2s_v2 en **spaincentral** (cada una con SOLO sus datos; agregador co-alojado en `vm-silo-grande` por cuota/capacidad de estudiante), entrenamiento por gRPC. **FedAvg y FedProx corrieron sobre las 3 VMs reales, el `val_loss` por ronda se logueó en vivo a Azure ML Studio (dashboard) y el modelo quedó registrado (`tfm-federado-global v1`).** El val_loss federado converge de ~4,9 a ~0,05 hacia la ronda 3-5, igual que la simulación documentada. `flower_app/` (Flower App de despliegue, reutiliza `ClienteSilo`/`MLPConEmbeddings` de `src/`) + `infra/` (12 scripts). Se resolvió en directo una cadena larga de fricciones reales del nivel estudiante (región bloqueada, cuota, capacidad, encoding, consistencia eventual, bug de `az role assignment`, Python 3.10→3.11, detach de demonios por SSH, emoji en cp1252, env que no llega al ServerApp...), todas dejadas reproducibles en los scripts. Coste real <$5; VMs **deallocated** al terminar. Ver RESEARCH_LOG Sesión 30 y 30b.

## ▶️ PRÓXIMA TAREA — revisión de la tutora

**Todas las fases del proyecto están cerradas** (T1-T3 incluida T3.5, Fase 4 MLOps, simulacro en
Azure y Fase 4b de negocio). La memoria del TFM está redactada y compila.

- `memoria/main.pdf` — 59 páginas, 8 capítulos + anexos, 12 figuras, 8 tablas (Sesión 33: se
  eliminó el capítulo de Presupuesto — no es requisito de la normativa de MADI y era la única
  parte del documento apoyada en cifras estimadas sin trazabilidad a `reports/`). Se genera con
  `cd memoria && bash compilar.sh`.
- Borrador listo para enviar a la tutora (Idoia Ochoa) y recoger correcciones.

**Pendiente de la revisión:**
1. Incorporar las correcciones que indique la tutora.
2. Añadir el logotipo oficial de Tecnun a la portada si se dispone de él
   (`memoria/figuras/logo_tecnun.png`; la línea está comentada en `portada.tex`).
3. Opcional: capturar el cuadro de mando de Azure ML antes de destruir la infraestructura.

**Líneas futuras identificadas** (documentadas en el capítulo de conclusiones): privacidad
diferencial (RQ5, la más relevante), selección por serie entre modelo personalizado y local,
ajuste de hiperparámetros específico de la personalización, validación con varios orígenes.

## ⏳ Pendiente de decisión / acción del usuario

- **Regenerar el token de Kaggle** (se pegó en un chat; higiene). Al hacerlo, actualizar `C:\Users\alefl\.kaggle\access_token`.
- **Simulacro federado en Azure (Sesión 30/30b) — YA EJECUTADO con éxito.** La infra queda en `rg-tfm-federado` con las 3 VMs **deallocated** (gasto ~0). Para retomarla: `az vm start -g rg-tfm-federado --ids $(az vm list -g rg-tfm-federado --query '[].id' -o tsv)` y re-arrancar la federación (`infra/04_start_federation.sh`). Para borrar todo y liberar el crédito por completo: `bash infra/destroy.sh`. **Pendiente opcional:** capturar una captura de pantalla del dashboard (Azure ML Studio → experimento `tfm-federado-simulacro`) para la memoria, antes de `destroy`.
- **Docker Desktop** — confirmar que el daemon local está activo para verificar `docker run` (el build ya se verificó con éxito en la Sesión 29).
- **Cuenta de Weights & Biases** (gratuita, [wandb.ai](https://wandb.ai)) — al crearla, ejecutar
  `wandb login` y luego `wandb sync wandb/<carpeta>` para subir la corrida ya registrada en local.
  No bloquea la Fase 3 (funciona en modo offline mientras tanto).

## 🧭 Cómo ejecutar (recordatorio)

```bash
cd "/c/Users/alefl/OneDrive/Escritorio/tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/<script>.py
```
