# RESEARCH_LOG.md — Cuaderno de laboratorio

> Registro cronológico y minucioso de cada paso de la investigación: método exacto, comandos
> reproducibles, resultados con cifras precisas, interpretación y justificación de cada decisión.
> Este documento es la materia prima de los capítulos de Metodología y Resultados del TFM —
> se escribe con el mismo estándar con el que se documentaría un experimento para publicación.
>
> **Convención:** cada entrada es inmutable una vez cerrada. Si algo se corrige después, se añade
> una entrada nueva que referencia y corrige la anterior — nunca se edita el historial.
>
> Complementa a `docs/STATE.md` (qué toca hacer ahora) y `docs/PLAN.md` (el plan completo).
> Este archivo responde: **qué se hizo, con qué método, qué se obtuvo, y por qué se decidió lo que se decidió.**

---

## Sesión 1 — Configuración del entorno y del proyecto

**Fecha:** 2026-07-09
**Objetivo:** disponer de un entorno de trabajo reproducible y un repositorio versionado antes de tocar ningún dato.

### Método

1. Verificación del entorno base: `python --version` → Python 3.11.9; `git --version` → 2.35.1; `pip` operativo.
2. Creación de la estructura de directorios del proyecto en `C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado\`:
   ```
   src/  data/raw/  data/processed/  notebooks/  reports/figures/  configs/  docs/
   ```
3. Inicialización de repositorio git (`git init`).
4. Creación de `.gitignore` — excluye credenciales (`*.json` salvo `configs/`, `kaggle.json`, `access_token`, `.env`), datos (`data/raw/`, `data/processed/`, `*.csv`, `*.parquet`), entorno (`.venv/`, `__pycache__/`) y artefactos de experimentos (`wandb/`, `mlruns/`, `*.ckpt`).
5. Entorno virtual: `python -m venv .venv`.
6. Instalación de dependencias base (`requirements.txt`): `kaggle`, `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`.

### Resultado

Versiones instaladas confirmadas: `kaggle 2.2.3`, `pandas 3.0.3`, `numpy 2.4.6`, `matplotlib 3.11.0`, `seaborn 0.13.2`, `scipy 1.17.1`, `scikit-learn 1.9.0`.

### Autenticación con Kaggle

Se generó un token de API de Kaggle (formato nuevo `KGAT_...`) y se guardó en `C:\Users\alefl\.kaggle\access_token`. **Nota de seguridad:** el token fue compartido inicialmente por el usuario en el chat de la sesión; se marcó como pendiente su regeneración (ver `STATE.md`), práctica estándar cuando un secreto ha quedado expuesto en un canal no cifrado de forma persistente.

Verificación de autenticación:
```bash
./.venv/Scripts/kaggle.exe competitions list --search "store-sales" --page-size 5
```
→ Devuelve la lista de competiciones correctamente, confirmando autenticación válida.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

---

## Sesión 2 — Adquisición de datos

**Fecha:** 2026-07-09
**Objetivo:** obtener el dataset base (Corporación Favorita) de forma reproducible y documentada.

### Método

Descarga vía CLI oficial de Kaggle de la competición `store-sales-time-series-forecasting`:
```bash
./.venv/Scripts/kaggle.exe competitions download -c store-sales-time-series-forecasting -p data/raw
```
Requisito previo: el usuario debe haber aceptado las reglas de la competición en `kaggle.com/competitions/store-sales-time-series-forecasting/rules` — sin este paso, la API devuelve error 403.

Descompresión: `unzip -o -q store-sales-time-series-forecasting.zip`.

### Resultado — inventario de archivos obtenidos

| Archivo | Filas (datos) | Contenido |
|---|---|---|
| `train.csv` | 3.000.888 | Ventas diarias, `id, date, store_nbr, family, sales, onpromotion` |
| `stores.csv` | 54 | Metadatos de tienda: `store_nbr, city, state, type, cluster` |
| `transactions.csv` | 83.488 | Nº de transacciones por tienda y día |
| `oil.csv` | 1.218 | Precio diario del petróleo WTI (`dcoilwtico`) |
| `holidays_events.csv` | 350 | Festivos y eventos: `date, type, locale, locale_name, description, transferred` |
| `test.csv` / `sample_submission.csv` | 28.512 | No usados (formato de competición Kaggle, no del diseño experimental propio) |

Tamaño total del zip descargado: 21,4 MB.

### Nota metodológica

`test.csv` y `sample_submission.csv` pertenecen al formato original de la competición Kaggle y **no se usan** en este TFM — nuestro propio corte de validación temporal (walk-forward) se construye a partir de `train.csv` en la Fase 1, ya que necesitamos control total sobre las fechas de corte para la comparación entre condiciones experimentales.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
./.venv/Scripts/kaggle.exe competitions download -c store-sales-time-series-forecasting -p data/raw
cd data/raw && unzip -o -q store-sales-time-series-forecasting.zip
```

---

## Sesión 3 — Análisis exploratorio inicial

**Fecha:** 2026-07-09
**Script:** `src/01_explore.py`
**Objetivo:** caracterizar la estructura de tiendas y ventas, y obtener una primera estimación de si un reparto geográfico en silos produce heterogeneidad real entre ellos.

### Método

1. Carga de `stores.csv` (54 filas) y `train.csv` (3.000.888 filas, parseando `date`).
2. Estadística descriptiva de tiendas: nº de estados/provincias, ciudades, tipos, clusters; distribución de tiendas por estado.
3. Estadística descriptiva de ventas: rango de fechas, nº de familias de producto, ventas totales, participación de las principales familias.
4. Propuesta inicial de partición en 3 silos por agrupación geográfica de provincias (Pichincha → Silo Quito; Guayas → Silo Guayaquil; resto → Silo Otras Regiones).
5. Cálculo de heterogeneidad entre silos mediante **divergencia de Jensen-Shannon** sobre la distribución de ventas por familia de producto, y **ratio de escala** (venta media acumulada por tienda).

### Resultados

**Estructura de tiendas** — 54 tiendas en 16 provincias y 22 ciudades; 5 tipos (A–E); 17 clusters.

Distribución por provincia: Pichincha 19, Guayas 11, Santo Domingo de los Tsáchilas 3, Azuay 3, Manabí 3, Cotopaxi 2, Tungurahua 2, Los Ríos 2, El Oro 2, y 8 provincias con 1 tienda cada una.

**Estructura de ventas** — rango temporal: 2013-01-01 a 2017-08-15 (1.688 días). 33 familias de producto. Ventas totales acumuladas: 1.073.644.952 unidades.

Participación de las principales familias en el total de ventas:

| Familia | % del total |
|---|---|
| GROCERY I | 32,0 % |
| BEVERAGES | 20,2 % |
| PRODUCE | 11,4 % |
| CLEANING | 9,1 % |
| DAIRY | 6,0 % |
| BREAD/BAKERY | 3,9 % |
| POULTRY | 3,0 % |
| MEATS | 2,9 % |

**Partición geográfica propuesta (provisional):** Silo Quito (19 tiendas), Silo Guayaquil (11 tiendas), Silo Otras Regiones (24 tiendas).

**Heterogeneidad de mezcla de categorías (Jensen-Shannon, 0 = idéntico, 1 = máxima divergencia):**

| Par de silos | JS |
|---|---|
| Quito vs. Guayaquil | 0,0026 |
| Quito vs. Otras Regiones | 0,0082 |
| Guayaquil vs. Otras Regiones | 0,0050 |

**Heterogeneidad de escala (venta media acumulada por tienda):** Quito 30.793.021 · Guayaquil 15.014.036 · Otras Regiones 13.475.965 (ratio máx/mín ≈ 2,3×).

Figura generada: `reports/figures/01_silos_overview.png`.

### Interpretación

La partición geográfica produce una diferencia de **escala** clara (~2,3×) pero una diferencia de **comportamiento de compra** (mezcla de categorías) muy pequeña — los valores de JS (0,003–0,008) son bajos en términos absolutos. En Ecuador, las proporciones de gasto por categoría son similares entre regiones: la heterogeneidad "interesante" para el problema de federated learning no reside principalmente en la geografía. **Esta observación motivó la Sesión 4.**

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/01_explore.py
```

---

## Sesión 4 — Comparación de estrategias de partición en silos

**Fecha:** 2026-07-09
**Script:** `src/02_silo_strategy.py`
**Objetivo:** decidir con evidencia cuantitativa, no por intuición, cómo repartir las 54 tiendas en los 3 silos que simularán operadores de supermercado independientes.

### Método

Tres estrategias de partición, todas en 3 grupos: (1) por **región**; (2) por **formato de tienda** (tipos A–E ordenados por venta real y agrupados en Grande/Mediano/Pequeño); (3) por **cluster** (los 17 clusters de Favorita agrupados en 3 terciles por venta media).

Para cada estrategia, tres métricas de heterogeneidad entre silos: **JS media de mezcla de categorías** (comportamiento de compra), **ratio de escala** (tamaño de negocio), **diferencia de intensidad promocional** (rango de la tasa media de `onpromotion`).

### Resultados

| Estrategia | Nº silos | JS mezcla (comportamiento) | Ratio escala | Diferencia promo | Tiendas por silo |
|---|---|---|---|---|---|
| Región | 3 | 0,0053 | 2,29× | 0,605 | Guayaquil 11 · Otras 24 · Quito 19 |
| **Formato** | 3 | **0,0166** | 2,21× | 0,618 | Grande 27 · Mediano 8 · Pequeño 19 |
| Cluster | 3 | 0,0104 | 3,23× | 0,755 | C1 23 · C2 19 · C3 12 |

Orden real de los 5 tipos de tienda por venta media (mayor → menor): A > D > B > E > C.

### Interpretación

La partición **por formato de tienda** produce la mayor heterogeneidad de comportamiento de compra (JS = 0,0166, más del triple que por región), manteniendo diferencia de escala (2,2×) y promocional comparables. La partición por cluster tiene la mayor heterogeneidad de escala (3,2×) pero es la más difícil de justificar narrativamente. **Limitación detectada:** el reparto provisional por formato deja el silo "Mediano" con solo 8 tiendas — desequilibrio a corregir en la tarea T1.1.

### Decisión tomada

**Se adopta la partición por formato de tienda** (tipos A–E agrupados en Grande/Mediano/Pequeño), presentados como análogos a tres formatos reales de supermercado que coexisten en un mismo mercado. **Justificación:** (1) máxima heterogeneidad de comportamiento — la dimensión más relevante para el valor del federado; (2) narrativa de negocio clara; (3) heterogeneidad de escala suficiente para justificar la personalización. **Estado:** propuesta por el asistente, *pendiente de confirmación explícita del autor del TFM* antes de construir el pipeline (registrado en `STATE.md`).

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/02_silo_strategy.py
```

---

## Sesión 5 — Diseño detallado del pipeline de modelado y del algoritmo federado

**Fecha:** 2026-07-11
**Objetivo:** fijar, a bajo nivel, la variable objetivo, el conjunto de features, la arquitectura del modelo, el bucle de entrenamiento local y el algoritmo de FedAvg — respondiendo explícitamente a por qué se incluye o excluye cada elemento.

### Variable objetivo

Misma para las 5 condiciones experimentales (A–E): `y(tienda, familia, semana) = Σ ventas diarias de esa semana`. Se predice en escala `log(1+y)` y se deshace con `exp(ŷ)−1` al final.

**Justificación de la transformación logarítmica (con detalle matemático):**

- **Motivo 1 — heterocedasticidad.** La varianza de las ventas crece con su media (tiendas grandes fluctúan en miles de unidades, tiendas pequeñas en decenas). MSE asume varianza constante; log(1+y) estabiliza la varianza.
- **Motivo 2 — robustez a valores atípicos.** El terremoto de abril de 2016 (evento documentado en el propio dataset) genera un pico extremo de ventas que, en escala cruda, dominaría el entrenamiento por MSE. En escala logarítmica su influencia se comprime.
- **Ejemplo numérico verificado:** dos tiendas con el mismo 20% de error relativo (10.000→12.000 unidades vs. 50→60 unidades) contribuyen **4.000.000 vs. 100** al error cuadrático en escala cruda (40.000× de diferencia), pero **0,0333 vs. 0,0321** en escala log(1+y) — prácticamente idénticos. Esto confirma que log(1+y) aproxima el **error relativo**, no el absoluto: para desviaciones pequeñas, `log(y)−log(ŷ) ≈ (y−ŷ)/y`.
- **Validación externa:** la competición Kaggle original de este mismo dataset (Corporación Favorita Grocery Sales Forecasting, 2017) usó como métrica oficial **NWRMSLE** (Normalized Weighted Root Mean Squared Logarithmic Error), exactamente la familia de métrica que resulta de entrenar con MSE sobre log(1+y) — confirmando que el propio diseño del dataset anticipó este mismo problema. Fuente: documentación de la competición, recuperada 2026-07-10 ([kaggle.com/c/favorita-grocery-sales-forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting); ponderación 1,25 para productos perecederos, 1,00 para el resto).
- **Consistencia con la métrica de evaluación:** al entrenar en log-espacio, el objetivo de optimización queda alineado con **WMAPE** (la métrica de evaluación ya elegida) — evita el desajuste metodológico de entrenar para una cosa y medir otra.
- **Relevancia específica para el sistema federado:** los 3 silos tienen escalas de venta distintas (ratio ~2,2×, Sesión 4). Entrenar en escala cruda produciría redes que aprenden "en unidades" distintas por silo, dificultando que el promediado de FedAvg tenga sentido. Log(1+y) normaliza la escala numérica **antes** de la agregación, haciendo la mecánica de FedAvg más estable.
- **Matiz honesto:** la aproximación log-diferencia ≈ error relativo es válida para desviaciones pequeñas; se degrada si el modelo predice muy mal. Para el rango de error esperado en este proyecto es una aproximación razonable, no exacta.
- **Distinción de un concepto relacionado pero distinto:** esto NO es lo mismo que el *log loss* (entropía cruzada) usado en clasificación — ese log viene de la verosimilitud de una distribución categórica; el de aquí es una transformación del objetivo de regresión para estabilizar varianza.

### Variables de entrada (features)

Ver tabla completa en el documento técnico de la sesión (9 grupos: autorregresivas — lags 1/2/4/8 semanas y medias móviles —, calendario, días de pago, promoción, festivos, petróleo, embeddings de familia y de tienda/cluster). Todas calculadas solo con información anterior a la semana predicha.

**Decisión — precio excluido de las features:** `train.csv` de Favorita no contiene ninguna columna de precio (solo `onpromotion`, un indicador binario/contador de si el producto está en oferta). Es una **limitación real del dataset**, no una elección de diseño: el modelo puede aprender el efecto de estar en promoción, pero no la elasticidad-precio. Se documentará explícitamente en el capítulo de limitaciones de la memoria.

### Arquitectura del modelo

MLP con embeddings de entidad — misma arquitectura para las 5 condiciones (A–E); lo que cambia entre condiciones es qué datos la entrenan, no su forma.

- Entrada: ~12 features continuas + embedding de familia (33→8 dim) + embedding de tienda/cluster (≤54→8 dim) = 28 dimensiones.
- Densa(28→64) → ReLU → Dropout(0,2) → Densa(64→32) → ReLU → Densa(32→1).
- ~10.000–15.000 parámetros.
- Pérdida: Huber (robusta a outliers) sobre el objetivo en log(1+y). Optimizador: Adam, lr inicial 1e-3.

**Justificación de no usar LightGBM/XGBoost como modelo federado:** FedAvg promedia pesos numéricos entre participantes; las matrices de una red neuronal se prestan a esto de forma directa, mientras que la estructura de árboles de un GBM no se promedia de forma limpia (existe investigación de "federated boosting", fuera de alcance). LightGBM se mantiene como referencia clásica no-federada.

**Justificación de no usar LSTM/TCN (de entrada):** con datos semanales (no diarios) y series moderadamente cortas (~241 semanas), una red recurrente añade complejidad de entrenamiento y de defensa sin garantía de mejora clara. Se documenta como extensión posible (stretch), no como necesidad — en línea con la instrucción explícita de evitar sobre-ingeniería.

### El algoritmo de FedAvg (McMahan et al., 2017)

Participantes del federado: **los 3 silos** (no las 54 tiendas sueltas). Cada silo centraliza internamente sus propias tiendas (legítimo, es el mismo operador simulado); entre silos, solo se intercambian pesos.

```
servidor inicializa pesos globales w⁰
para cada ronda t = 1...T:
    servidor envía wᵗ a los 3 silos
    cada silo k entrena localmente E épocas desde wᵗ con sus propios datos → wₖᵗ⁺¹
    cada silo envía wₖᵗ⁺¹ al servidor (nunca datos crudos)
    servidor agrega: wᵗ⁺¹ = Σₖ (nₖ/n) · wₖᵗ⁺¹        # media ponderada por nº de filas
    servidor evalúa pidiendo a cada silo su métrica local (un escalar, no datos)
hasta que la métrica de validación deje de mejorar
```

Jerarquía de 3 niveles: (1) tienda — personalización opcional post-federado (condición E); (2) silo — participante real de FedAvg; (3) global — modelo resultante de la agregación.

### Validación temporal y evaluación

Walk-forward (test = últimas ~8 semanas por silo, nunca se mezcla con el futuro). WMAPE por serie/silo/agregado. Test de Wilcoxon pareado entre condiciones.

### Estado de las decisiones de esta sesión

Todas presentadas como propuesta técnica razonada; arquitectura y algoritmo son la base de referencia para implementar en la Fase 1-4 del `PLAN.md`. Ajustables si la experimentación real (Fase 2 en adelante) sugiere cambios — se documentará como nueva sesión si ocurre.

---

## Sesión 6 — Notebook de repaso: regresión lineal desde cero

**Fecha:** 2026-07-11
**Notebook:** `notebooks/00_refresher_regresion_lineal.ipynb`
**Objetivo:** material de apoyo pedagógico (no un resultado de investigación) — repasar los fundamentos de regresión lineal simple y múltiple implementándolos sin librerías de ML, como preparación conceptual antes de construir el MLP de la Sesión 5. Se documenta aquí porque es parte del trabajo del proyecto y su ejecución generó resultados verificables.

### Método

1. Instalación de herramientas de notebook en el entorno: `nbformat`, `nbclient`, `nbconvert`, `ipykernel`, `jupyter_client` (añadidas a `requirements.txt`).
2. Registro del entorno virtual como kernel de Jupyter: `python -m ipykernel install --user --name tfmfl --display-name "Python (tfm-forecasting-federado)"`.
3. Generación programática del notebook (vía script con `nbformat`, no escrito a mano en la UI de Jupyter) con 25 celdas: teoría (derivación matemática de las ecuaciones normales, simple y múltiple), datos sintéticos con relación verdadera conocida, implementación manual, descenso de gradiente, visualizaciones, y verificación final contra `scikit-learn`.
4. Ejecución completa del notebook con `jupyter nbconvert --execute`, embebiendo las salidas (gráficas y resultados) directamente en el archivo.

### Resultados

**Regresión simple** — datos sintéticos (n=60, relación verdadera y=3+2x+ruido):
- Ecuaciones normales: β₀=3,1773, β₁=1,9256
- Descenso de gradiente (lr=0,03, 3.000 iteraciones): β₀=3,1773, β₁=1,9256 — **coincide hasta el 4º decimal**
- scikit-learn (`LinearRegression`): β₀=3,177304, β₁=1,925616 — **coincide exactamente** con la implementación manual
- R² = 0,9165

**Regresión múltiple** — datos sintéticos (n=120, 2 variables, relación verdadera y=5+1,5x₁−2x₂+ruido):
- Ecuación normal matricial: β=[4,6575, 1,4925, −1,7920]
- scikit-learn: β=[4,657499, 1,492499, −1,792031] — **coincide exactamente**
- R² = 0,9477

0 errores de ejecución en las 13 celdas de código.

### Interpretación

La coincidencia exacta entre la implementación manual (ecuaciones normales), el descenso de gradiente, y `scikit-learn` confirma que ambas derivaciones matemáticas (Sección 1.3 y 3.2 del notebook) están correctamente implementadas. El ajuste del learning rate/iteraciones de descenso de gradiente (de lr=0,01/500 iter a lr=0,03/3.000 iter en una segunda iteración de este mismo notebook) fue necesario para alcanzar convergencia completa — un recordatorio práctico de que el descenso de gradiente, a diferencia de la ecuación cerrada, depende de hiperparámetros bien elegidos. Este mismo problema (elegir learning rate y nº de pasos) reaparecerá al entrenar el MLP de la Sesión 5.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
./.venv/Scripts/python.exe -m ipykernel install --user --name tfmfl --display-name "Python (tfm-forecasting-federado)"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --ExecutePreprocessor.kernel_name=tfmfl notebooks/00_refresher_regresion_lineal.ipynb --output 00_refresher_regresion_lineal.ipynb
```

---

## Sesión 7 — Reenfoque tras reunión con el tutor: equilibrio técnico + negocio

**Fecha:** 2026-07-14
**Tipo:** decisión de alcance (no ejecución de código)
**Objetivo:** registrar el reenfoque acordado con el tutor del TFM y su justificación.

### Contexto

El autor se reunió con el tutor experto en federated learning para revisar el enfoque del
proyecto. El tutor indicó que el TFM debe equilibrar la vertiente técnica y la de negocio:
no basta con demostrar que el modelo federado supera a las alternativas locales/centralizadas —
hay que demostrar que supera a los **métodos convencionales de forecasting usados en retail**, y
argumentar **por qué tiene valor de negocio** superior a alternativas ya establecidas como los
**data clean rooms**.

### Investigación de apoyo (previa a fijar la decisión)

Se investigó el estado 2026 de los data clean rooms frente al federated learning para
fundamentar la comparación con fuentes reales, no por intuición:

- Los data clean rooms (Snowflake, AWS, Decentriq, etc.) están diseñados para **responder
  preguntas sobre datos existentes** (medición de campañas, audiencias) mediante joins
  controlados — no para **entrenar modelos predictivos** de forma nativa. El dato normalmente
  **sí se mueve** a un entorno compartido (aunque el output que sale sea agregado), y requiere
  confiar en un operador o en hardware de cómputo confidencial.
- El federated learning, en cambio, es un **método de entrenamiento** en el que el dato nunca
  sale de su origen — encaja de forma directa con la tarea de este TFM (construir un modelo
  predictivo entre operadores que no comparten datos).
- Nota honesta: ambos mundos están **convergiendo** (algunos clean rooms ya incorporan
  entrenamiento federado internamente) — no es una dicotomía absoluta, y el TFM debe reconocerlo
  en vez de presentar el FL como sustituto universal.
- Fuentes: [Federated Learning vs Data Clean Rooms (Sherpa.ai)](https://sherpa.ai/blog/federated-learning-vs-data-clean-rooms-2/), [Snowflake Data Clean Rooms](https://www.snowflake.com/en/product/features/data-clean-rooms/), [Data Clean Rooms 2026: guía de decisión](https://www.digitalapplied.com/blog/data-clean-rooms-advertising-2026-marketer-decision-guide) — consultadas 2026-07-14.

### Decisión tomada

1. **Preguntas de investigación reformuladas** — de 3 (todas técnicas) a 5 (2 técnicas + 2 de
   negocio + 1 stretch). Ver `PLAN.md` sección 0 para el texto exacto (RQ1-RQ5).
2. **Baselines de la Fase 2 ampliados** (nueva tarea T2.2b): se añaden suavizado exponencial
   (ETS/Holt-Winters) y LightGBM por tienda como referencia de "métodos convencionales de
   retail" — sin esto, RQ2 ("¿supera el federado a lo convencional?") no tendría con qué
   contrastarse más allá de un naive.
3. **Dos capítulos nuevos** (Fase 4b, `PLAN.md`): comparación estructurada FL vs. data clean
   rooms (T4b.1), y cuantificación del valor de negocio en euros a partir de la mejora de WMAPE
   (T4b.2). Ambos son análisis/escritura — no requieren entrenar modelos nuevos.
4. **Estado del arte de la memoria** pasa a cubrir tres bloques: métodos de forecasting +
   federated learning + data clean rooms (antes solo cubría FL).

### Impacto en el cronograma

Mínimo. Los dos capítulos nuevos son de análisis y redacción (ejecutables incluso sin acceso a
los datos, p. ej. desde una sesión móvil); el único coste técnico añadido es entrenar ETS y
LightGBM como baselines, rápido comparado con el resto del pipeline. Se trata sobre todo de un
reequilibrio de la narrativa de la memoria, no de más experimentación.

### Pendiente de esta sesión

Confirmar con el autor si además de ETS quiere incluir ARIMA/Prophet en los baselines
convencionales (pregunta abierta, sin resolver a fecha de esta entrada).

---

## Sesión 8 — Overview completo de datos: verificación exhaustiva + criterio de silo validado

**Fecha:** 2026-07-14
**Objetivo:** (1) validar con datos reales si "venta media" es el criterio correcto para agrupar
los tipos de tienda en silos, comparándolo con una alternativa (tráfico de clientes); (2)
inspeccionar a fondo los 6 ficheros del dataset, verificando cifras exactas en vez de asumirlas
de sesiones anteriores, y producir un diccionario de datos permanente (`docs/DATA.md`).

### Método — validación del criterio de silo

Se calculó, por tipo de tienda (A-E), la venta total media por tienda y la media diaria de
transacciones (`transactions.csv`), y se comparó el orden resultante de cada métrica.

### Resultados

| Tipo | Venta media/tienda | Transacciones/día |
|---|---|---|
| A | 39.227.094 | 3.104,3 |
| D | 19.504.628 | 1.593,7 |
| B | 18.157.579 | 1.698,0 |
| E | 14.955.609 | 1.179,0 |
| C | 10.962.316 | 1.021,6 |

Correlación de Pearson (venta total vs. transacciones medias, a nivel de tienda): **0,9096**.
Orden por venta: A > D > B > E > C. Orden por transacciones: **A > B > D > E > C** (D y B
intercambian posición).

### Interpretación y decisión

D y B son prácticamente indistinguibles (su orden depende de qué métrica se use), mientras que A
queda claramente por encima y C claramente por debajo **en ambas métricas**. La frontera natural
{A} | {D,B} | {E,C} — la "Propuesta 2" de la sesión anterior — es **robusta a la elección de
variable**, lo que la valida con más solidez que un solo criterio. Se documenta también, con
honestidad metodológica, que la variable "ideal" en un escenario con todos los datos disponibles
sería superficie de venta (m²) y nº de referencias — atributos de diseño de la tienda, no
proxies derivados de venta/tráfico — pero no están disponibles en este dataset.

**Decisión: se confirma la Propuesta 2 de partición de silos** (Grande=tipo A · Mediano=tipos D+B
· Pequeño=tipos E+C), pendiente de generar `stores_silos.csv` en la siguiente tarea.

### Método — overview exhaustivo de datos

Inspección directa (dtypes, nulos, rangos, duplicados, completitud del panel) de los 6 ficheros.
Resultado completo en `docs/DATA.md` (nuevo documento permanente). Hallazgos nuevos no
documentados en sesiones anteriores:

- **8 tiendas abrieron durante el periodo del dataset** (no las 54 desde 2013-01-01): tiendas
  36, 53, 20, 29, 21, 42, 22, y **52 (con solo ~4 meses de historia real, abrió 2017-04-20)**.
  Repartidas entre tipos: A=1, B=2, C=1, D=2, E=2 — **el tipo E tiene 2 de sus 4 tiendas (50%)
  con historial corto**, lo que puede subestimar levemente su venta media reportada arriba. No
  cambia la decisión de silo (E ya quedaba en el grupo "Pequeño" en ambas propuestas), pero se
  documenta como limitación honesta.
- **`transactions.csv` tiene huecos**: 7.340 combinaciones tienda-día faltantes de las 90.828
  esperadas (tiendas cerradas ese día).
- **`oil.csv` tiene 43 nulos** (días sin cotización bursátil) — se resuelve solo al agregar a
  semana (T1.2), sin necesitar interpolación explícita.
- **`holidays_events.csv`**: 12 festivos "transferred" (la fecha real del festivo está en un
  registro `type=Transfer` aparte) y 38 fechas con más de un evento simultáneo — ambos casos
  exigen tratamiento explícito en el feature engineering de T1.2/T1.4, no se pueden ignorar.

### Impacto en el plan

Se añade a T1.2/T1.4 (`PLAN.md`) el recorte del histórico de cada tienda a su fecha real de
apertura, y el tratamiento correcto de festivos transferidos/duplicados — ambos ya anotados
en `docs/DATA.md` sección 2.6 y sección 4 (plan de uso por fase).

### Reproducibilidad
Comandos completos ejecutados y verificados en esta sesión — ver historial de shell; resumen de
resultados en `docs/DATA.md`.

---

## Sesión 9 — Confirmaciones del autor: silos, baselines, y justificación de granularidad

**Fecha:** 2026-07-15
**Tipo:** decisiones de diseño confirmadas + cierre de T1.1
**Objetivo:** registrar, con su justificación completa (material directamente reutilizable en
la memoria), las tres decisiones que el autor confirmó tras revisar la evidencia de las
Sesiones 7-8.

### Decisión 1 — Partición de silos: Propuesta 2, confirmada

El autor confirma la Propuesta 2 (Grande=tipo A · Mediano=tipos D+B · Pequeño=tipos E+C),
validada en la Sesión 8 por dos métricas independientes (venta total y tráfico de clientes,
correlación 0,91, ambas coinciden en la misma frontera natural). Ejecutado en
`src/04_generate_silos.py`, que genera `data/processed/stores_silos.csv`. Reparto final:

| Silo | Tipos | Nº tiendas |
|---|---|---|
| Grande | A | 9 |
| Mediano | D, B | 26 |
| Pequeño | E, C | 19 |

**T1.1 queda cerrada.**

### Decisión 2 — Baselines de la Fase 2 (T2.2b): LightGBM + ETS/Holt-Winters, sin ARIMA/Prophet

Confirmado tras la investigación de la Sesión 7/8: la evidencia de 2026 muestra que LightGBM
con features temporales es "una apuesta segura por defecto" en producción, y que ETS/Holt-Winters
es "uno de los modelos más desplegados en la industria, particularmente en planificación de
demanda de retail" — mientras que ARIMA y Prophet quedan sistemáticamente por detrás en
benchmarks recientes. Añadir ARIMA/Prophet no aportaría más rigor a RQ2 y sí más tiempo de
desarrollo, con el plazo de &lt;3 meses ya comprometido por el reenfoque de negocio (Sesión 7).

### Decisión 3 — Granularidad: familia × semana, no SKU × día (justificación para la memoria)

**Contexto y honestidad metodológica:** en una discusión previa a la descarga de datos, se
había planteado el nivel de SKU como preferible para el valor de negocio (permite decisiones
de surtido producto a producto, p. ej. qué referencia concreta descontinuar). Al inspeccionar
el dataset realmente descargado (`store-sales-time-series-forecasting`, la versión "Getting
Started" que Kaggle deriva del Favorita original de 2017) se confirmó que **no incluye
`item_nbr`** — solo 33 familias agregadas (`docs/DATA.md`, sección 2.1). La competición
original de 2017 sí tenía nivel de ítem (~4.000 productos), pero es ~40 veces más pesada
(~125M filas) y no es la que se descargó. Esta sección documenta, con conocimiento de causa,
la decisión de **no** volver a cambiar de dataset y mantener familia × semana.

**Justificación (cuatro argumentos):**

1. **Ortogonalidad con las preguntas de investigación centrales.** RQ1 y RQ2 preguntan si el
   *mecanismo de entrenamiento* (federado vs. local/centralizado/convencional) mejora la
   previsión — esa pregunta se responde igual de bien a nivel de familia que de SKU. La
   granularidad del objetivo no determina la respuesta sobre si federar ayuda.
2. **Coste y riesgo de alcance.** Pasar a SKU multiplicaría por ~120 el número de series por
   tienda (33 familias → ~4.000 ítems) e introduciría demanda intermitente (muchos ceros por
   producto y día), un problema de investigación distinto (métodos tipo Croston) fuera del
   alcance de este TFM. Con &lt;3 meses de plazo, es un riesgo real de no completar ni la
   comparación federada ni los capítulos de negocio ya comprometidos (Fase 4b).
3. **Calidad de la señal y rigor de la comparación.** A nivel familia-semana la serie es mucho
   menos ruidosa que SKU-día; menos ruido de fondo permite atribuir con más confianza (mayor
   poder estadístico en el test de Wilcoxon) las diferencias de WMAPE entre condiciones al
   mecanismo de entrenamiento, y no a varianza intrínseca de series dispersas.
4. **Relevancia de negocio de la semana como ciclo.** La planificación de pedidos y personal en
   retail es mayoritariamente semanal, no diaria — la granularidad temporal elegida coincide
   con el ciclo de decisión real de un gerente de tienda, no es solo una simplificación técnica.

**Limitación reconocida y mitigación (a incluir explícitamente en el capítulo de limitaciones):**
se pierde la capacidad de decisión a nivel de producto individual. Mitigación: la arquitectura
(embedding de familia, Sesión 5/10) es estructuralmente el mecanismo de "categoría como
respaldo estadístico" que permitiría extender el sistema a SKU en trabajo futuro — un producto
nuevo heredaría el embedding de su familia como punto de partida (shrinkage). El capítulo de
valor de negocio (T4b.2) debe matizar que la cifra en € calculada corresponde a decisiones de
reposición agregada por familia, no a decisiones de surtido SKU a SKU.

### Estado de las tres decisiones

**Cerradas y confirmadas por el autor (2026-07-15).** No quedan preguntas abiertas de la
Sesión 7. Próxima tarea: T1.2 (pipeline de datos de modelado).

---

## Sesión 10 — Notebook de apoyo: MLP + Embeddings + FedAvg desde cero

**Fecha:** 2026-07-15
**Notebook:** `notebooks/01_refresher_mlp_embeddings_fedavg.ipynb`
**Objetivo:** material de apoyo pedagógico (no resultado de investigación) — construir con
código real y ejecutado las tres piezas de la arquitectura del proyecto (MLP, embeddings,
FedAvg) por separado, para entender cada una antes de implementarlas sobre los datos reales.

### Método

1. Instalación de `torch`, `lightgbm`, `statsmodels` en el entorno (añadidas a `requirements.txt`,
   necesarias también para T1.2 en adelante y para T2.2b).
2. Generación programática del notebook (24 celdas) con `nbformat`, en 4 secciones:
   - **MLP desde cero** (numpy, forward+backward manual, un hidden layer): regresión lineal vs.
     MLP sobre datos sintéticos con relación no lineal (`y = sin(x)·2 + 0.3x + ruido`).
   - **Embeddings** (PyTorch `nn.Embedding`, 2D para visualización directa): 6 categorías
     sintéticas con "efecto verdadero" no ordinal, visualización del espacio aprendido.
   - **MLP + embeddings juntos** (PyTorch): arquitectura de juguete con la misma forma que la
     Sesión 5 (features continuas + embedding de tienda + embedding de familia).
   - **FedAvg simulado**: datos sintéticos con 3 silos de escala muy distinta (imitando
     Grande/Mediano/Pequeño real: 40/18/8), comparación explícita local (A) vs. federado (D).
3. Ejecución completa con `jupyter nbconvert --execute`.

### Resultados

- **MLP vs. regresión lineal:** MSE 0,086 (MLP) vs. 1,633 (regresión) — **94,7% menos error**
  en datos con relación no lineal a propósito.
- **Embedding:** pérdida de entrenamiento 26,65 → 0,18. Verificación visual: las categorías con
  efecto verdadero similar (BEVERAGES, DAIRY, BREAD, efecto real ≈4,5-5,0) quedaron próximas en
  el espacio de embedding 2D, sin habérselo indicado al modelo explícitamente.
- **FedAvg (25 rondas), MSE en validación conjunta (mezcla de los 3 silos):**

| Modelo | MSE validación global |
|---|---|
| Local — Grande | 122,91 |
| Local — Mediano | 209,54 |
| Local — Pequeño | 534,75 |
| **Federado** | **123,53** |

0 errores de ejecución en las 14 celdas de código.

### Interpretación

El resultado es más matizado (y más honesto) que "el federado gana siempre": el modelo
federado **empata con el mejor de los tres modelos locales** (Grande, la diferencia de 0,6 es
ruido) y **reduce drásticamente el error frente a los otros dos** (41% menos que Mediano, 77%
menos que Pequeño). Lectura para la memoria: ningún silo sabe de antemano si será el "afortunado"
(como Grande, cuyo modelo local generaliza sorprendentemente bien) o el "desafortunado" (como
Pequeño); federar ofrece la garantía del mejor caso sin el riesgo del peor — un argumento de
**reducción de riesgo**, no solo de precisión media, que es directamente reutilizable en el
capítulo de valor de negocio (T4b.2) y en la comparación con data clean rooms (T4b.1).

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --ExecutePreprocessor.kernel_name=tfmfl notebooks/01_refresher_mlp_embeddings_fedavg.ipynb --output 01_refresher_mlp_embeddings_fedavg.ipynb
```

---

## Sesión 11 — T1.2: construcción del dataset de modelado

**Fecha:** 2026-07-15
**Script:** `src/05_build_modeling_dataset.py`
**Objetivo:** construir el dataset tienda×familia×semana que alimentará todas las condiciones
experimentales (A-F), con las covariables y correcciones ya identificadas en `docs/DATA.md`.

### Método

1. **Recorte por apertura tardía:** se elimina, por tienda, todo registro anterior a su primera
   venta &gt; 0 (las 8 tiendas identificadas en la Sesión 8).
2. **Agregación:** `groupby(store_nbr, family, week_start)` — semana = lunes ISO de cada fecha.
   `ventas` = suma; `onpromotion` = media; se guarda además `dias_con_dato` (nº de días reales
   agregados esa semana, para detectar semanas parciales en los bordes).
3. **Unión con `stores_silos.csv`** (silo, ciudad, provincia, tipo).
4. **Petróleo:** calendario diario completo → `ffill`+`bfill` de los huecos (fines de semana
   bursátiles) → media semanal.
5. **Festivos**, con el tratamiento decidido en `docs/DATA.md`: se excluyen los festivos con
   `transferred=True` (no se observan esa fecha) y el tipo `Work Day` (recuperación de puente,
   no es festivo). Flags separados por nivel: `es_festivo_nacional` (aplica a todas las filas
   de esa semana), `es_festivo_regional` (cruce por `state`), `es_festivo_local` (cruce por `city`).
6. **Calendario:** `semana_del_anio`, `mes`, y `semana_con_dia_pago` (la semana contiene el
   día 15 o el último día del mes — efecto de nómina documentado en Ecuador, Sesión 3).

### Resultados

| Verificación | Resultado |
|---|---|
| Filas eliminadas por apertura tardía | 222.057 (7,40% del total) |
| Filas finales | 399.762 (54 tiendas × 33 familias × hasta 242 semanas) |
| Reparto por silo | Grande 64.482 · Mediano 192.687 · Pequeño 142.593 |
| Nulos en el dataset final | **0** (en las 17 columnas) |
| Venta total original vs. dataset final | **1.073.644.952 = 1.073.644.952 (idéntica)** |
| Filas con festivo nacional / regional / local | 107.646 / 957 / 7.755 |
| Filas en semana con día de pago | 184.899 (46,25%) |
| Semanas parciales (`dias_con_dato` &lt; 7) | 10.197 (2,55%) — bordes del dataset o de la apertura de cada tienda |

### Interpretación

La igualdad exacta entre la venta total original y la del dataset agregado confirma que el
recorte por apertura tardía solo eliminó filas con venta 0 (por construcción, ya que se filtra
todo lo anterior a la primera venta &gt;0) — **no se perdió ninguna unidad de venta real**, es
una verificación de integridad, no una casualidad. La proporción de semanas con día de pago
(46,25%) es coherente con la aritmética esperada (~2 días de pago al mes sobre ~4,3 semanas/mes).

### Decisión pendiente para T1.4 (no se resuelve aquí)

Las semanas parciales (2,55% de las filas) quedan señaladas vía `dias_con_dato` pero **no se
excluyen ni se ponderan todavía** — se decidirá en T1.4 (feature engineering) si se descartan,
se ponderan por `dias_con_dato/7` en la función de pérdida, o se dejan tal cual.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/05_build_modeling_dataset.py
```

---

## Sesión 12 — Qué significa realmente `type` (honestidad sobre una variable no documentada)

**Fecha:** 2026-07-15
**Objetivo:** documentar con precisión el significado (y los límites de lo que se conoce) de la
variable `type` de `stores.csv`, usada como base de la partición en silos.

### Método

Búsqueda de la documentación oficial de Kaggle/Corporación Favorita sobre `stores.csv`, contraste
con análisis exploratorios independientes de la comunidad, y verificación propia de si `cluster`
(sí documentado como "agrupación de tiendas similares") anida dentro de `type` mediante tabla
cruzada `pd.crosstab(type, cluster)`.

### Resultados

- **La documentación oficial de Kaggle no define `type`.** Solo describe `cluster` explícitamente.
  `type` es una etiqueta de una letra (A-E) sin criterio de asignación revelado por Corporación
  Favorita ni por Kaggle.
- **Consenso de la comunidad** (múltiples EDA independientes en Kaggle/Medium): `type` se interpreta
  como una proxy de tamaño/formato de tienda, inferido empíricamente — no confirmado oficialmente.
  Los tipos A y D concentran el mayor volumen de ventas; el tipo E es el menos frecuente (4 tiendas).
  Coincide con el orden hallado de forma independiente en la Sesión 4 (A>D>B>E>C).
- **Anidamiento `cluster` dentro de `type`:** 16 de los 17 clusters caen íntegramente dentro de un
  único `type` (solo el cluster 10 mezcla tipos D y E, mayoritariamente E). Evidencia de que
  `cluster` es una subdivisión más fina *dentro* de `type`, no una clasificación independiente.

### Interpretación y decisión

Se documenta `type` en la memoria como **una clasificación de formato/tamaño de tienda no revelada
oficialmente**, cuyo uso en este TFM se justifica no por confiar ciegamente en la etiqueta, sino
porque **se validó de forma independiente con datos propios** (venta media y tráfico de clientes,
Sesión 8) antes de construir los silos sobre ella. Se añade el hallazgo del anidamiento con
`cluster` como evidencia adicional de que `type` captura una estructura real y estable, no arbitraria.

---

## Sesión 13 — T1.3: corte temporal walk-forward + exclusión de semanas parciales

**Fecha:** 2026-07-15
**Script:** `src/06_temporal_split.py`
**Objetivo:** particionar el dataset en train/val/test por fecha (walk-forward, sin aleatoriedad),
y cerrar la decisión pendiente de la Sesión 11 sobre las semanas parciales de los bordes.

### Decisión — semanas parciales: EXCLUIDAS

**Confirmado por el autor (2026-07-15).** Se eliminan las 10.197 filas (2,55% del dataset) con
`dias_con_dato < 7` — semanas incompletas en los bordes del histórico de cada tienda (por el
límite del dataset o por el recorte de apertura tardía, Sesión 11). **Justificación para la
memoria:** una semana con, p. ej., solo 2 días de datos no es comparable a una semana completa
— su venta agregada infravalora sistemáticamente la demanda real de esa semana, introduciendo
un sesgo (no ruido aleatorio) que ninguna de las condiciones experimentales podría corregir por
sí sola. Excluirlas es preferible a ponderarlas, por simplicidad y porque es solo un 2,55% de
las filas — no se pierde una fracción relevante de información.

### Método — corte temporal

1. Sobre el dataset ya sin semanas parciales, se define el corte por **fecha global** (misma
   frontera para las 3 silos — el corte particiona por tiempo, no por silo):
   **test = últimas 8 semanas completas · val = las 8 anteriores · train = el resto.**
2. Verificación de ausencia de fuga: `max(train.week_start) < min(val.week_start)` y
   `max(val.week_start) < min(test.week_start)`.
3. Verificación de cobertura por silo en val/test (que ningún silo se quede sin representación
   suficiente en los conjuntos de evaluación).

### Resultados

| Split | Desde | Hasta | Nº semanas | Filas |
|---|---|---|---|---|
| Train | 2013-01-07 | 2017-04-17 | 220 | 361.053 |
| Val | 2017-04-24 | 2017-06-12 | 8 | 14.256 |
| Test | 2017-06-19 | 2017-08-07 | 8 | 14.256 |

Verificación formal de fronteras: **train &lt; val &lt; test, sin solape** (aserción automática en
el script, superada).

**Cobertura por silo en val y test** — coincide de forma exacta con nº_tiendas × 33 familias × 8
semanas para los tres silos (Grande 9×33×8=2.376 ✓ · Mediano 26×33×8=6.864 ✓ · Pequeño
19×33×8=5.016 ✓), confirmando que **ninguna tienda falta** en el periodo de evaluación reciente
— si alguna tienda hubiera cerrado antes de agosto de 2017, el recuento no habría cuadrado.

Total: 389.565 filas conservadas (97,4% de las 399.762 de T1.2).

### Salidas

- `data/processed/dataset_modelado.parquet` — dataset final con columna `split`.
- `configs/split_config.json` — fechas de corte exactas, para reproducibilidad y para que
  cualquier script posterior (baselines, condiciones A-F) lea el mismo corte sin recalcularlo.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/06_temporal_split.py
```

---

## Sesión 14 — Reporte de progreso (Fase 1 completa)

**Fecha:** 2026-07-15
**Script:** `src/07_report_figures.py`
**Objetivo:** cierre de la Fase 1 (Pipeline de datos) con un reporte de progreso estilo memoria,
para comunicar el estado del proyecto y dejar material gráfico reutilizable directamente en la
memoria final. No es una sesión de investigación nueva — consolida y visualiza las Sesiones 7-13.

### Material generado

Cinco figuras a partir de datos reales del proyecto (no maquetas), guardadas en `reports/figures/`:

- `r1_serie_semanal_split.png` — ventas semanales agregadas 2013-2017, con el pico del terremoto
  de abril de 2016 señalado y las bandas de validación/test marcadas.
- `r2_composicion_silos.png` — nº de tiendas y venta media por silo (Propuesta 2).
- `r3_venta_vs_transacciones.png` — venta media y tráfico por tipo de tienda, evidencia visual del
  salto natural que justifica la frontera Grande={A}.
- `r4_type_cluster_heatmap.png` — tabla cruzada type×cluster, evidencia del anidamiento (Sesión 12).
- `r5_fedavg_toy_resultado.png` — resultado de la simulación de FedAvg del notebook 01 (Sesión 10).

Estas figuras son directamente reutilizables en los capítulos de Metodología y Resultados de la
memoria final — no son ilustrativas, están calculadas sobre los datos verificados del proyecto.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/07_report_figures.py
```

---

## Sesión 15 — Estado del arte de la partición en silos (resumen para la memoria) + uso de `cluster`

**Fecha:** 2026-07-18

### Qué hace la literatura de FL para retail (resumen conciso)

- **Ningún paper revisado usa cadenas competidoras reales como silos** — el dato no existe público (es
  información comercialmente sensible que ninguna empresa publica).
- **PA-CFL** (2025, el más comparable a este TFM: FL para forecasting de ventas retail heterogéneo) usa
  **un único dataset de una empresa (DataCo)**, particionado en **14 regiones geográficas** como
  "clientes" federados — no 14 empresas distintas.
- **Conclusión:** particionar un único dataset real por un eje que genere heterogeneidad controlada
  **es la metodología estándar y aceptada del campo**, no una simplificación exclusiva de este TFM.
  Fuente: [PA-CFL (arXiv 2503.12220)](https://arxiv.org/html/2503.12220v2).

### Nuestra decisión y por qué es válida

Se parte Corporación Favorita (una cadena) por **formato de tienda** en 3 silos — mismo principio que
PA-CFL (partir por región), pero con una diferencia a nuestro favor: **probamos primero partir por
región** (el eje "de libro") y encontramos heterogeneidad casi nula (JS=0,005, Sesión 4); el formato fue
la alternativa que sí dio heterogeneidad real y medible. Nuestra elección está, por tanto, respaldada
por evidencia propia, no solo por convención.

### Qué haría falta en un contexto de vida real

En un despliegue real, los silos serían **cadenas** (Mercadona, DIA, Alcampo), cada una con formatos
mezclados internamente; el formato pasaría de ser el eje del silo a ser **una variable/embedding más**
del modelo. El mecanismo de FedAvg no cambia: solo cambia qué frontera de datos representa cada
participante. Esta transferibilidad se documenta como limitación reconocida, no como fallo de validez.

### `cluster`: por qué no se usa como variable de entrenamiento

17 clusters sobre 54 tiendas → media de 3,18 tiendas/cluster, **4 clusters con una sola tienda** (5, 12,
16, 17), 7 de 17 con ≤2 tiendas. Un embedding necesita varios miembros por categoría para generalizar;
con esta dispersión, la mayoría de clusters no aportarían señal compartida, solo memorización de casos
individuales. **Uso decidido:** no como input de entrenamiento; sí como **validación cualitativa
posterior** (Fase 3/6) — proyectar el embedding de tienda aprendido por el modelo (PCA 2D) y colorear
por `cluster`, para comprobar si el modelo redescubre por su cuenta la agrupación que Favorita ya había
hecho con criterios no revelados (evidencia externa de que el modelo aprende estructura real).

---

## Plantilla para futuras entradas

```markdown
## Sesión N — <título>

**Fecha:** AAAA-MM-DD
**Script:** `src/NN_nombre.py` (si aplica)
**Objetivo:** <una frase>

### Método
<pasos exactos>

### Resultados
<cifras exactas, tablas>

### Interpretación
<qué significan los resultados>

### Decisión tomada (si aplica)
<qué se decidió y por qué>

### Reproducibilidad
```bash
<comando exacto>
```
```
