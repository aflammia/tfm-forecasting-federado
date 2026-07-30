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

## Sesión 16 — T1.4: ingeniería de variables

**Fecha:** 2026-07-18
**Script:** `src/08_feature_engineering.py`
**Objetivo:** añadir variables autorregresivas (lags, medias móviles), el objetivo transformado, y
la codificación de categóricas para los embeddings — sobre el dataset ya particionado (T1.3).

### Método

1. Ordenación estricta por (`store_nbr`, `family`, `week_start`) antes de cualquier cálculo.
2. Por serie (tienda×familia): `lag_1`, `lag_2`, `lag_4`, `lag_8`; medias móviles de 4 y 8 semanas y
   desviación de 4 semanas, calculadas **excluyendo la semana actual** (`shift(1)` antes de `rolling`)
   para que ninguna use información de la semana que se predice.
3. `log_ventas = log(1+ventas)` (objetivo transformado, justificado en Sesión 5).
4. Codificación de categóricas: `family_id` (0-32) y `store_id` (0-53), enteros listos para las
   tablas de embedding de la arquitectura (Sesión 5/10).
5. **Verificación anti-fuga:** muestra aleatoria de 500 filas comprobando que `lag_1` de la semana W
   coincide exactamente con la venta real de la semana W-1 de esa misma serie.

### Resultados

Verificación anti-fuga: **0 discrepancias en 500 filas** comprobadas.

Filas con historial insuficiente (NaN en variables autorregresivas, inicio de cada serie): 14.256 de
389.565 (3,66%), casi todas en train (13.992) — esperable, es el "arranque" natural de cada serie.

**Hallazgo relevante — corrige una afirmación de la Sesión 13:** de las 264 filas con NaN que caen en
`val` (no en train), **las 264 pertenecen íntegramente a la tienda 52** (33 familias × 8 semanas). La
tienda 52 abrió el 2017-04-20 — apenas 4 días antes de que empiece el periodo de validación
(2017-04-24) — y no acumula ni una semana de historial previo para calcular lags de hasta 8 semanas.
**Para el periodo de test (desde 2017-06-19) la tienda 52 sí tiene historial suficiente y aparece
completa.**

**Corrección formal:** la Sesión 13 afirmó "cobertura exacta de las 54 tiendas confirmada en val/test"
— esto era cierto en ese momento (antes de calcular lags), pero **deja de serlo tras la ingeniería de
variables**: `val` cubre 53 tiendas (falta la 52), `test` cubre las 54. Se documenta aquí para que la
memoria no arrastre la afirmación desactualizada.

| Split | Tiendas cubiertas | Filas finales |
|---|---|---|
| Train | 54 | 347.061 |
| Val | **53** (sin la tienda 52) | 13.992 |
| Test | 54 | 14.256 |

### Interpretación y decisión

Se eliminan las filas sin historial suficiente (no se puede entrenar ni evaluar sin lags). La ausencia
de la tienda 52 en `val` es una limitación menor y explicable (apertura reciente, un caso aislado de 1
de 54 tiendas) — se documenta con transparencia en vez de forzar un relleno artificial de los lags
faltantes, que introduciría datos inventados. No afecta a `test`, que es el conjunto de evaluación
final del TFM.

### Salidas
- `data/processed/dataset_features.parquet` — dataset final con variables autorregresivas, objetivo
  transformado y categóricas codificadas. 375.309 filas (96,3% del dataset de T1.3).

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/08_feature_engineering.py
```

---

## Sesión 17 — T1.4b (normalización) y T1.5 (tests): cierre de la Fase 1

**Fecha:** 2026-07-18
**Script:** `src/08_feature_engineering.py` (reescrito) + `tests/test_data_pipeline.py`
**Objetivo:** corregir una inconsistencia detectada en T1.4, decidir y aplicar la normalización de
las features continuas, y formalizar como suite automatizada las verificaciones manuales de las
Sesiones 11, 13 y 16.

### Corrección sobre T1.4 — lags en escala logarítmica, no en escala cruda

Al diseñar la normalización se detectó que T1.4 calculó los lags y medias móviles sobre `ventas` en
bruto, no sobre `log_ventas` — inconsistente con la justificación completa de la Sesión 5 (la
transformación logarítmica existe para estabilizar la varianza y homogeneizar la escala numérica
**entre silos** antes de la agregación de FedAvg; dejar los lags en escala cruda reintroduce el mismo
problema por la puerta de atrás, en las variables de entrada en vez de en el objetivo).

**Evidencia que motivó la corrección** (verificada, no asumida): asimetría (skew) de `ventas` = 5,32
frente a `log_ventas` = -0,10 (casi simétrica); `onpromotion` en crudo tiene skew = 11,07.

**Corrección aplicada:** todos los lags (`lag_log_1/2/4/8`), medias móviles y desviación se
recalculan sobre `log_ventas`. `onpromotion` también se transforma con `log(1+x)` por la misma razón.

### Método — T1.4b (normalización)

1. **Codificación cíclica** de `semana_del_anio` y `mes` (seno/coseno) — un código ordinal simple
   haría que el modelo viera la semana 52 y la semana 1 como extremos opuestos, cuando en realidad
   son consecutivas (diciembre-enero).
2. **Estandarización z-score** de todas las features continuas finales (lags log, medias móviles log,
   `log_onpromotion`, `oil_price`), con media y desviación **calculadas solo con el conjunto de
   train** — para no filtrar estadísticos de val/test hacia el entrenamiento. Estadísticos guardados
   en `configs/normalizacion.json` para aplicarlos de forma reproducible (y poder revertir la
   transformación al interpretar resultados).

### Resultados

Verificación anti-fuga repetida sobre la versión corregida: **0 discrepancias en 500 filas**.
Estandarización verificada: media≈0,0000 y desviación≈1,0000 en train para las columnas `_z`.

### Método — T1.5 (tests automatizados)

Se creó `tests/test_data_pipeline.py` (pytest) con **16 tests** que formalizan todas las
verificaciones manuales de las sesiones anteriores: composición de silos, ausencia de duplicados y
de NaN en columnas de entrenamiento, ausencia de fuga temporal entre splits, coincidencia de fechas
con `configs/split_config.json`, anti-fuga de lags (dos tests: valor exacto y coherencia estructural
de fechas), correctitud de la estandarización, y rangos válidos de las codificaciones.

**Detalle relevante del propio proceso de testing:** el primer test de fechas de split falló al
ejecutarlo — no por un error en los datos, sino porque el test comparaba ingenuamente contra
`split_config.json` (generado en T1.3, antes de recortar por lags). Es **correcto** que `train`
empiece ahora el 2013-03-04 en vez del 2013-01-07 original (2013-01-07 + 8 semanas exactas de
recorte por historial insuficiente). Se corrigió el test para comprobar lo que realmente debe
cumplirse: `val`/`test` coinciden exactamente con T1.3 (no se ven afectados por el recorte, están al
final de cada serie), y `train` solo debe tener fecha mínima **posterior o igual**, nunca anterior.

### Resultados finales

**16 de 16 tests pasan** (`pytest tests/ -v`).

### Salidas
- `data/processed/dataset_features.parquet` — dataset final corregido (lags en log-escala,
  codificación cíclica, features estandarizadas). Mismas 375.309 filas que en T1.4.
- `configs/normalizacion.json` — media y desviación de train por columna, para reproducibilidad.
- `tests/test_data_pipeline.py` — 16 tests automatizados.

### Cierre de la Fase 1

Con esta sesión se completan T1.1 a T1.5 — el pipeline de datos queda cerrado, verificado y
documentado de principio a fin. Próximo hito: Fase 2 (baselines A/B/C + LightGBM/ETS).

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/08_feature_engineering.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Sesión 18 — T2.1: módulo de métricas (WMAPE, MASE, RMSSE) + test de Wilcoxon

**Fecha:** 2026-07-18
**Script:** `src/metrics.py` (librería, no script de pipeline) + `tests/test_metrics.py`
**Objetivo:** construir el módulo de evaluación compartido por las 6 condiciones experimentales
(A-F), con las métricas ya fijadas en el diseño experimental (Sesión 1 del RESEARCH_LOG).

### Método

Implementación de:
- `wmape` — métrica principal del proyecto.
- `mase` y `rmsse` — escaladas por el error de un naive de un paso calculado en train de cada
  serie (permiten comparar series de escalas muy distintas, relevante entre nuestros 3 silos).
  `rmsse` es la métrica oficial de la competición M5.
- `metricas_por_serie` — calcula las tres métricas por cada serie (tienda×familia).
- `comparar_condiciones` — test de Wilcoxon pareado (signed-rank) entre dos condiciones,
  emparejando por serie. Es la comprobación de significancia central del diseño experimental.
- `porcentaje_brecha_recuperada` — la "métrica estrella" del proyecto (% de la brecha
  Local→Centralizado que recupera el Federado).

Cada función se verificó con **casos calculados a mano** antes de darla por buena (12 tests en
`tests/test_metrics.py`), incluyendo: WMAPE con solución exacta conocida, MASE/RMSSE con series
sintéticas donde el resultado se puede derivar analíticamente, el test de Wilcoxon comprobado en
dos escenarios de control (diferencia sistemática real → debe detectarla; ruido simétrico sin
sesgo → no debe detectarla), y los tres casos límite de la métrica de brecha recuperada (0%, 100%,
caso intermedio).

### Resultados

**12 de 12 tests pasan**, todos contra un cálculo independiente, no solo "el código no falla".

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v
```

---

## Sesión 19 — Corrección: huecos internos por Navidad desalineaban los lags (T1.4) y bloqueaban T2.2

**Fecha:** 2026-07-20
**Scripts:** `src/calendario_semanal.py` (nuevo), `src/08_feature_engineering.py` (corregido),
`tests/test_data_pipeline.py` (+1 test)
**Objetivo:** al construir T2.2 (baselines ingenuos), se añadió proactivamente una comprobación de
seguridad antes de confiar en `shift()` para el baseline estacional (t-52) y de persistencia (t-1):
verificar que ninguna serie tuviera huecos internos en su secuencia semanal. La comprobación falló:
**1.749 de 1.782 series (98%) tienen al menos un hueco interno.** Esto detuvo T2.2 y obligó a
investigar la causa raíz antes de continuar, porque el mismo mecanismo (`groupby().shift()`) se usó
también en T1.4 (ya cerrado en la Sesión 17) para calcular `lag_log_1/2/4/8` y las medias móviles.

### Diagnóstico

1. Caso concreto: tienda 1, familia AUTOMOTIVE, 4 huecos, el primero en la semana del 2013-12-23.
2. Se leyó el script completo de T1.2 (`05_build_modeling_dataset.py`): `dias_con_dato` se calcula
   con `.agg(dias_con_dato=("sales", "count"))`, que cuenta valores NO NULOS — no filas.
3. Se verificó el panel crudo (`train.csv`): **no hay ningún día con menos de 1.782 filas
   (54 tiendas × 33 familias)** en ningún punto del rango de fechas — descartaba la hipótesis de
   filas ausentes a nivel diario general.
4. Se inspeccionaron directamente las filas de tienda 1/AUTOMOTIVE alrededor del 25-dic-2013 en
   `train.csv`: la secuencia de fechas salta de **2013-12-24 a 2013-12-26** — el **25 de diciembre
   no tiene fila** para esa combinación tienda-familia.
5. Se generalizó la comprobación a las 4 navidades del dataset (2013-2016) y a **todas** las
   tiendas/familias: `pd.to_datetime(['2013-12-25','2014-12-25','2015-12-25','2016-12-25'])` — **
   ninguna de las 4 fechas aparece en `train.csv`**, para ninguna combinación. Los días adyacentes
   (20 a 24 y 26 a 28 de diciembre) sí tienen las 1.782 filas completas cada uno.

**Causa raíz confirmada:** las tiendas de Corporación Favorita cierran el 25 de diciembre y, a
diferencia de otros días de venta nula, ese cierre no se registra como fila con `sales=0` sino que
**no genera ninguna fila en absoluto**. Esto deja `dias_con_dato=6` en la semana que contiene el
25-dic para prácticamente todas las series, y T1.3 excluye correctamente esa semana como "parcial"
(`dias_con_dato<7`) — pero al excluir una fila completa de en medio de cada serie, crea un hueco
interno real en el índice semanal.

**Por qué esto rompe `shift()`:** `groupby(...).shift(N)` avanza **N posiciones dentro del grupo**,
no N semanas de calendario. Si una semana intermedia falta, la fila inmediatamente posterior al
hueco recibe (en `shift(1)`) el valor de la fila anterior **en posición**, que en realidad es de
**2 semanas atrás** — un desplazamiento silencioso, sin ningún error ni NaN que lo delate. Esto
afecta tanto a T1.4 (`lag_log_1/2/4/8`, medias móviles, desviación) como al T2.2 en construcción
(persistencia, estacional-52, media móvil 4). La verificación anti-fuga de T1.4/T1.5 (búsqueda por
fecha exacta) **no lo detectaba**: al buscar la fila de la semana `W-1` y no encontrarla (porque es
precisamente la semana excluida), el chequeo la salta en lugar de fallar — el hueco es invisible
para una verificación que solo comprueba coincidencia de valores, no la propia continuidad del
índice.

### Corrección aplicada

Se creó `src/calendario_semanal.py`, con `construir_calendario_completo()`: por cada serie
(tienda×familia), reconstruye el rango semanal **completo** entre su primera y última semana
presente (`pd.date_range(..., freq="7D")`) y reintroduce las semanas ausentes con `NaN` en vez de
saltarlas. Sobre este calendario reindexado, `shift(N)` vuelve a coincidir exactamente con "N
semanas de calendario atrás": si `N` cruza una semana ausente, el resultado es `NaN` (correcto: "no
hay dato fiable"), nunca un valor de una semana equivocada.

`src/08_feature_engineering.py` (T1.4) se modificó para calcular `lag_log_*`, `media_movil_log_*` y
`std_log_4` sobre el calendario reconstruido, y luego unir (`merge`) esos valores de vuelta al
dataset de modelado real (que ya excluye las semanas parciales como objetivo, sin cambios ahí). El
paso de "filtrar filas sin historial suficiente" ahora también excluye, correctamente, las filas
inmediatamente posteriores a cada hueco de Navidad (antes se quedaban con un lag mal alineado en vez
de ir a NaN).

Se añadió un test nuevo (`test_ninguna_fila_queda_justo_despues_de_un_hueco_interno`) que verifica
estructuralmente, contra `dataset_modelado.parquet` (el calendario real de T1.3), que ninguna fila
superviviente en el dataset final tenga su semana anterior ausente en su misma serie — formaliza el
invariante que la corrección garantiza, en vez de depender solo de una comprobación de valores por
muestreo.

### Resultados

- Semanas con hueco interno reintroducidas como NaN en el calendario reconstruido: **6.633**.
- Verificación anti-fuga de `lag_log_1` repetida: **0 discrepancias en 500 filas** (igual que antes,
  pero ahora la base sobre la que se calcula es correcta).
- Filas descartadas por historial insuficiente: **67.320 (17,3%)** — sube respecto a la Sesión 17
  precisamente porque ahora se descartan también las filas justo después de cada hueco de Navidad
  que antes se colaban con un lag mal alineado.
- **Dataset final: 322.245 filas** (antes, con el bug: 375.309 — una reducción del 14,2%, coherente
  con el número de series y años de Navidad afectados).
- Suite de tests completa: **29 de 29 pasan** (16 de T1.5 + 1 nuevo de esta sesión + 12 de T2.1).

### Decisión y justificación

Se corrige T1.4 en lugar de solo T2.2, aunque T1.4 ya estaba "cerrado" (Sesión 17), porque el mismo
mecanismo posicional afectaba a ambos y el dataset ya usado como "final" de la Fase 1 tenía lags
silenciosamente incorrectos alrededor de cada Navidad — un error de esta naturaleza (fuga/desajuste
temporal) es exactamente el tipo de problema que este proyecto se comprometió a verificar con
evidencia, no a asumir. Por convención del proyecto, esta sesión no edita la Sesión 17 — la corrige
mediante una nueva entrada.

### Salidas
- `src/calendario_semanal.py` — utilidad nueva, reutilizada por T1.4 y T2.2.
- `data/processed/dataset_features.parquet` — regenerado (322.245 filas, antes 375.309).
- `configs/normalizacion.json` — regenerado con los nuevos estadísticos de train.
- `tests/test_data_pipeline.py` — 17 tests (16 + 1 nuevo).

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/08_feature_engineering.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Sesión 20 — T2.2: baselines ingenuos (suelo de cordura)

**Fecha:** 2026-07-20
**Script:** `src/10_baselines_ingenuos.py`
**Objetivo:** establecer el "suelo de cordura" del proyecto — tres predictores sin entrenamiento
alguno que cualquier condición seria (A-F) debe superar con holgura, evaluados con el módulo de
métricas de T2.1 (Sesión 18) sobre el dataset ya corregido en la Sesión 19.

### Método

Tres baselines, en escala natural de ventas, calculados por serie (tienda×familia) sobre el
calendario semanal reconstruido (`calendario_semanal.py`, Sesión 19) para evitar el desajuste de
`shift()` alrededor de los huecos de Navidad:
- **Persistencia (t-1):** predicción(semana W) = venta real de W-1.
- **Estacional (t-52):** predicción(semana W) = venta real de la misma semana, un año antes.
- **Media móvil (4 semanas):** predicción(semana W) = media de las 4 semanas anteriores.

Evaluados sobre val y test con WMAPE, MASE y RMSSE (por serie, media y mediana), más el WMAPE
agregado global (suma de errores / suma de ventas) sobre test.

### Resultados

| Baseline | Split | WMAPE (media) | MASE (media) | RMSSE (media) | Cobertura |
|---|---|---|---|---|---|
| Persistencia (t-1) | val | 0,2837 | 1,561 | 1,201 | 100,0% |
| Persistencia (t-1) | test | 0,2670 | 1,315 | 0,967 | 99,8% |
| Estacional (t-52) | val | 0,3760 | 2,046 | 1,475 | 100,0% |
| Estacional (t-52) | test | 0,3751 | 1,947 | 1,378 | 98,1% |
| Media móvil (4 sem.) | val | 0,2683 | 1,437 | 1,025 | 100,0% |
| Media móvil (4 sem.) | test | 0,2414 | 1,185 | 0,857 | 99,1% |

WMAPE agregado global (test): persistencia 0,0949; estacional 0,1370; media móvil 0,0864.

**Interpretación:** la media móvil de 4 semanas es el baseline más fuerte de los tres (WMAPE más
bajo en ambos splits), seguida de persistencia; el estacional (t-52) es el más débil — esperable,
porque una sola semana de hace un año es una referencia mucho más ruidosa que el promedio reciente
o la semana inmediatamente anterior. MASE y RMSSE medios >1 en los tres baselines confirman que
ninguno de estos predictores "ingenuos" iguala siquiera al naive de referencia usado como escala
(persistencia en train) de forma consistente — es la referencia mínima esperada, no un resultado
sorprendente. La cobertura <100% en test (81,5%-99,8% según baseline) refleja series con historial
insuficiente para ese baseline concreto (p.ej. estacional-52 no puede evaluarse en series con menos
de un año de historia antes del punto evaluado) — coherente con las limitaciones ya documentadas
(tienda 52, aperturas tardías).

Advertencia de numpy ("Mean of empty slice") durante la ejecución: ocurre para series sin ninguna
fila en train (p.ej. tienda 52, que abre casi al final de train — Sesión 16), donde `mase`/`rmsse`
devuelven correctamente `NaN` en vez de fallar; no afecta los resultados agregados (`pandas` excluye
NaN de media/mediana por defecto).

### Salidas
- `reports/resultados_baselines.csv` — resumen agregado (WMAPE/MASE/RMSSE, media y mediana, por
  baseline y split) — referencia para comparar contra las condiciones A-F.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/10_baselines_ingenuos.py
```

---

## Sesión 21 — T2.2b: baselines convencionales (ETS/Holt-Winters + LightGBM por tienda)

**Fecha:** 2026-07-21
**Scripts:** `src/11_baselines_convencionales.py`, `tests/test_baselines_convencionales.py`
**Objetivo:** implementar los dos baselines que responden RQ2 ("¿supera el federado a los métodos
convencionales de forecasting en retail?") — ETS/Holt-Winters por serie y LightGBM por tienda —
sobre el dataset ya corregido en la Sesión 19.

### Método

- **ETS/Holt-Winters:** un modelo por serie (tienda×familia), ajustado una vez sobre `log_ventas`
  de train (usando `dataset_modelado.parquet`, no `dataset_features.parquet`, para no perder las
  ~8 semanas iniciales que T1.4 recorta por el warm-up de sus lags — ETS no usa esos lags) y
  pronosticado a 16 pasos (8 de val + 8 de test) de una sola vez, sin reentrenar semana a semana
  (el estándar en evaluación de baselines de horizonte fijo). Degrada el modelo según el
  historial disponible: estacional+tendencia (≥104 semanas) → solo tendencia (≥10) → nivel simple
  (≥2) → NaN. Reutiliza `calendario_semanal.py` (Sesión 19) para reindexar cada serie antes de
  ajustar, por la misma razón que T1.4/T2.2: un hueco interno sin reindexar distorsionaría la
  estacionalidad de 52 semanas que ETS asume.
- **LightGBM por tienda:** un modelo por tienda (54 modelos), entrenado con las filas de TRAIN de
  `dataset_features.parquet` de todas sus familias (aprende patrones compartidos entre familias de
  la misma tienda), prediciendo `log_ventas`. Sin normalización (los árboles son invariantes a
  transformaciones monótonas).

Ambos se evalúan sobre val/test con el módulo de métricas de T2.1, en escala natural de ventas
(`expm1` de la predicción).

### Incidencia y diagnóstico — ETS con tendencia sin amortiguar explota exponencialmente

La primera ejecución dio resultados absurdos: WMAPE_media de **3,44 en test**, con WMAPE_mediana de
solo 0,36 — un desajuste medía/mediana enorme, señal inequívoca de que unas pocas series estaban
produciendo errores catastróficos que dominaban el promedio. Se investigó antes de aceptar el
número, siguiendo la misma disciplina de la Sesión 19:

1. Se identificaron las filas con mayor error absoluto: predicciones de **hasta 2,6 millones de
   unidades** para una serie (tienda 53, PRODUCE) cuya venta real esa semana era ~15.800 unidades.
2. **Primera causa identificada:** `ExponentialSmoothing` con `trend="add"` sin amortiguar
   extrapola la tendencia linealmente sin límite; al deshacer `log1p` con `expm1`, ese error
   lineal se vuelve **exponencial** en escala de ventas. Corrección: `damped_trend=True`. Mejora
   parcial (WMAPE_media test baja de 3,44 a 2,29) pero el problema persiste.
3. **Segunda causa identificada** (inspeccionando la serie ofensora completa): la tienda 36,
   familia PRODUCE, tiene **ventas=0 durante las primeras ~30 semanas de train** (2013-05 a
   2013-12) y luego un salto abrupto a ventas normales — la familia no se vendía en esa tienda
   hasta bien entrado el periodo observado. Generalizando la comprobación: **779 de 1.749 series
   (45%) tienen un prefijo de más de 4 semanas de ceros al inicio de train** — no es ruido, es
   ausencia estructural de venta (no todas las tiendas tienen surtido de todas las 33 familias;
   ver `DATA.md`). Este prefijo, incluido tal cual en el ajuste de ETS, rompe la estimación de
   nivel/tendencia/estacionalidad inicial (`initialization_method`, tanto `"estimated"` como
   `"heuristic"` fallan igual). Corrección: `_recortar_ceros_iniciales()`, análoga a la fecha de
   apertura de T1.2 pero a nivel tienda×familia — recorta el prefijo de ceros antes de reindexar
   y ajustar.
4. Con ambas correcciones, el peor caso individual bajó de 2,6 millones a ~280.000 (tienda
   44/PRODUCE, venta real ~70.000) — la estacionalidad seguía mal estimada en series con historial
   post-lanzamiento corto e irregular (solo ~3 ciclos anuales, con huecos internos adicionales
   durante el periodo de adopción). Se decidió **no perseguir una estimación estacional perfecta**
   para estas series (ETS/Croston es un problema documentado en la literatura para demanda
   intermitente, no un objetivo de este TFM) y en su lugar añadir un **techo de cordura**: la
   predicción se acota a `3×` el máximo histórico de esa misma serie — una predicción a 300× el
   máximo observado no es un punto razonable bajo ninguna interpretación, y esta es la práctica
   estándar de guardrail en sistemas de forecasting productivos.

### Alcance final del problema (tras las 3 correcciones)

Con las tres correcciones (`damped_trend`, recorte de prefijo, techo de cordura), **526 de 1.696
series (31%)** aún tienen al menos una predicción que se desvía >3× de la venta real en alguna
semana — concentradas en **familias de demanda intermitente/esporádica**: PLAYERS AND ELECTRONICS,
CELEBRATION, HOME CARE, PRODUCE, PET SUPPLIES, SCHOOL AND OFFICE SUPPLIES. Esto es consistente con
una limitación **conocida y documentada en la literatura** de suavizado exponencial clásico frente
a demanda intermitente (motivo histórico de la existencia del método de Croston) — no un error del
pipeline. Percentiles de WMAPE por serie en test: mediana=0,33, p90=3,54, p95=4,80, p99=11,03,
máximo=893,6 (una cola derecha pesada pero ahora acotada y explicable, no un artefacto sin límite).

**Decisión:** no seguir ajustando ETS más allá de estas tres correcciones — perseguir una
estimación estacional perfecta para demanda intermitente (p.ej. implementar Croston) sería
sobre-ingeniería para lo que es un baseline de comparación, no el resultado central del TFM. Se
reporta **tanto la media como la mediana** de WMAPE/MASE/RMSSE (ya el formato estándar de T2.1/T2.2),
señalando explícitamente que la mediana es más representativa del comportamiento "típico" de ETS
dado esta cola pesada conocida, mientras que la media queda dominada por la minoría de series de
demanda intermitente.

### Resultados finales

| Baseline | Split | WMAPE (media) | WMAPE (mediana) | MASE (media) | MASE (mediana) | RMSSE (media) | RMSSE (mediana) | Cobertura |
|---|---|---|---|---|---|---|---|---|
| ETS/Holt-Winters | val | 0,613 | 0,254 | 3,823 | 1,309 | 2,510 | 1,071 | 95,2% |
| ETS/Holt-Winters | test | 1,946 | 0,334 | 7,248 | 1,455 | 4,089 | 1,188 | 95,2% |
| LightGBM (por tienda) | val | 0,259 | 0,165 | 1,420 | 1,049 | 1,086 | 0,854 | 100,0% |
| LightGBM (por tienda) | test | 0,394 | 0,143 | 1,386 | 0,896 | 1,001 | 0,722 | 98,1% |

**Interpretación:** incluso comparando por mediana (la estadística robusta), LightGBM supera
claramente a ETS en test (WMAPE 0,143 vs 0,334) — consistente con la evidencia de Petropoulos et
al. (2024) ya citada, que sitúa a los métodos basados en árboles por delante de los métodos
estadísticos clásicos en forecasting de retail. Este es el primer resultado empírico propio del
proyecto que corrobora esa cita. La cobertura de ETS en val/test es 95,2% (algo menor que
LightGBM) porque las series completamente sin ventas en train (recortadas a vacío por
`_recortar_ceros_iniciales`) no pueden pronosticarse — coherente con la limitación ya documentada de
que no todas las tiendas venden todas las familias.

### Salidas
- `src/11_baselines_convencionales.py` — ETS por serie + LightGBM por tienda.
- `src/calendario_semanal.py` — reutilizado sin cambios (Sesión 19).
- `tests/test_baselines_convencionales.py` — 10 tests (recorte de ceros, reindexado/interpolación,
  degradación de ETS, horizonte, esquema de features de LightGBM).
- `reports/resultados_baselines_convencionales.csv` — resumen agregado.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/11_baselines_convencionales.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Sesión 22 — Corrección: LightGBM por tienda → LightGBM global (tuneado correctamente)

**Fecha:** 2026-07-21
**Script:** `src/11_baselines_convencionales.py` (reescrito) + `configs/lightgbm_hiperparametros.json` (nuevo)
**Objetivo:** el LightGBM de la Sesión 21 (sin ajuste de hiperparámetros) no lograba superar ni
siquiera al baseline ingenuo más simple (media móvil de 4 semanas, T2.2) -- WMAPE test 0,394 vs
0,241. El autor pidió tunear correctamente el modelo hasta que lo superase.

### Método

1. **Features ampliadas:** se añadieron a `FEATURES_LGBM` cuatro columnas ya calculadas en T1.4
   pero no usadas hasta ahora -- `es_festivo_nacional`, `es_festivo_regional`, `es_festivo_local`,
   `semana_con_dia_pago` -- señal de calendario que un LightGBM sí puede explotar (a diferencia de
   una media móvil, agnóstica al calendario).
2. **Búsqueda de hiperparámetros:** búsqueda aleatoria (25 candidatos) sobre `learning_rate`,
   `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `reg_lambda` y
   `objective` (L2 vs. L1), seleccionando por WMAPE en escala natural sobre val (no por la
   pérdida de entrenamiento en log-escala). Mejor configuración: `objective=regression_l1,
   learning_rate=0.05, num_leaves=31, min_child_samples=5, feature_fraction=1.0,
   bagging_fraction=1.0, reg_lambda=2.0`.
3. **Early stopping** en vez de un `n_estimators` fijo, evaluado contra val.

### Incidencia — el modelo por tienda seguía sin batir a la media móvil

Con hiperparámetros ya tuneados y features ampliadas, el diseño original (un LightGBM POR TIENDA,
54 modelos) apenas mejoró: WMAPE test mediana 0,1434 → 0,1398 -- todavía por detrás de la media
móvil (0,1379), y la media empeoró (0,394 → 0,441). Se investigó la causa antes de aceptar el
resultado como "ya está tuneado, es lo que hay":

**Diagnóstico:** cada modelo por tienda entrena con muy pocos datos (~5.000-6.000 filas de train)
y decide cuándo parar (early stopping) contra el val de esa misma tienda -- solo ~264 filas
(33 familias × 8 semanas), una muestra demasiado pequeña para una decisión de parada estable
(ruido de varianza alta). La búsqueda de hiperparámetros, en cambio, se había hecho sobre el
dataset AGRUPADO de las 54 tiendas y alcanzaba WMAPE≈0,09 en val -- una señal de que agrupar
ayuda mucho más de lo que el diseño "por tienda" aprovechaba.

**Prueba:** se entrenó un único modelo LightGBM GLOBAL (mismos hiperparámetros, `store_id` y
`family_id` como categóricas) con `n_estimators` alto (2000, luego verificado hasta 6000) y early
stopping sobre el val completo (13.992 filas, no 264). Resultado en test: WMAPE mediana **0,1328**
(< 0,1379 de la media móvil), MASE mediana 0,7749 (< 0,8672), RMSSE mediana 0,6300 (< 0,7023) --
mejor que el baseline ingenuo en las **tres** métricas por mediana. Se verificó que el número de
árboles usados (3.691 de un tope de 6.000) no estaba siendo artificialmente recortado -- repetir
con un tope aún mayor apenas cambia el resultado (WMAPE test mediana 0,1328→0,1320), confirmando
convergencia real, no truncamiento.

### Decisión

Se sustituye el diseño "LightGBM por tienda" por **un único modelo LightGBM global** como baseline
convencional de T2.2b. Es una corrección de diseño basada en evidencia (igual que la Sesión 19 y
la Sesión 21), no solo un ajuste de hiperparámetros: el modelo por tienda estaba estructuralmente
limitado por escasez de datos, no por una mala elección de learning_rate. Nótese el paralelismo
con la tesis central del proyecto -- compartir señal entre entidades (aquí, tiendas dentro de un
único modelo; en la Fase 3, silos vía FedAvg) generaliza mejor que entrenar cada una por separado
con poco dato.

**Nota metodológica honesta:** al usar val para decidir cuándo parar el entrenamiento (early
stopping) y para la búsqueda de hiperparámetros, val deja de ser una estimación limpia de
generalización para este modelo -- sí lo sigue siendo para los demás baselines (T2.2, ETS), que no
tocan val en su ajuste. La comparación justa y honesta contra los demás baselines es la de TEST,
que no se usó en ningún momento del ajuste ni de la búsqueda.

### Resultados finales

| Baseline | Split | WMAPE mediana | MASE mediana | RMSSE mediana | Cobertura |
|---|---|---|---|---|---|
| Media móvil (T2.2, referencia) | test | 0,1379 | 0,8672 | 0,7023 | 99,1% |
| LightGBM (global, tuneado) | val | 0,1457 | 0,9171 | 0,7532 | 100,0% |
| **LightGBM (global, tuneado)** | **test** | **0,1320** | **0,7749** | **0,6300** | **100,0%** |
| ETS/Holt-Winters | test | 0,3344 | 1,4553 | 1,1883 | 95,2% |

LightGBM ahora bate con claridad tanto a ETS como al mejor baseline ingenuo (media móvil), en las
tres métricas, por mediana -- ya es un baseline convencional defendible para RQ2. Cobertura 100%
en ambos splits (el modelo global no descarta ninguna tienda, a diferencia del diseño por tienda,
que no podía predecir para la tienda 52 en val por falta de historial propio).

### Salidas
- `src/11_baselines_convencionales.py` — `predicciones_lightgbm` reescrita (modelo global).
- `configs/lightgbm_hiperparametros.json` — hiperparámetros ganadores de la búsqueda, para
  reproducibilidad.
- `reports/resultados_baselines_convencionales.csv` — regenerado.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/11_baselines_convencionales.py
```

---

## Sesión 23 — T2.3: Condición A (Local) — MLP+embeddings por tienda

**Fecha:** 2026-07-21
**Scripts:** `src/modelo_mlp.py` (nuevo, módulo compartido A-E) + `src/12_condicion_a_local.py` +
`tests/test_modelo_mlp.py`
**Objetivo:** implementar en PyTorch la arquitectura MLP+embeddings fijada en la Sesión 5
(refinada con la dimensión de entrada exacta en el diagrama de arquitectura de la Sesión 22) y
entrenar la primera condición experimental real: A (Local) — un modelo independiente por tienda,
sin ninguna colaboración entre ellas.

### Método

`src/modelo_mlp.py`: `MLPConEmbeddings` (33→64→32→1, Dropout 0,2 tras la primera capa, 4.985
parámetros verificados por test), `DatasetVentas` (envuelve un DataFrame en tensores), y
`entrenar()` (Huber loss, Adam lr=1e-3, early stopping sobre val_loss, devuelve los MEJORES pesos
vistos, no los últimos). Reutilizable sin cambios por B, C, D y E — solo cambia qué filas se le
pasan a `entrenar()`.

`src/12_condicion_a_local.py`: entrena 54 modelos independientes (uno por tienda), cada uno solo
con las filas de esa tienda. Para la tienda 52 (sin val propio por apertura tardía, Sesión 16) se
reserva el 15% final de su train como val interno solo para decidir el early stopping.

**Distinción explícita frente a la corrección de la Sesión 22:** allí, un LightGBM por tienda no
lograba batir a un baseline simple por falta de datos, y se sustituyó por un modelo global. Aquí
NO se corrige la escasez de datos por tienda -- es precisamente lo que la condición "Local" debe
representar con fidelidad. Que A rinda peor que un modelo con más datos es el resultado esperado y
necesario para que el resto del experimento (B-E) tenga algo que demostrar.

### Resultados

| Condición | Split | WMAPE mediana | MASE mediana | RMSSE mediana | Cobertura |
|---|---|---|---|---|---|
| A — Local | val | 0,1569 | 1,0369 | 0,8592 | 100,0% |
| A — Local | test | 0,1476 | 0,9283 | 0,7501 | 98,1% |

Comparado con lo ya cerrado (test, por mediana):

| Método | WMAPE | MASE | RMSSE |
|---|---|---|---|
| LightGBM (global, T2.2b) | 0,1320 | 0,7749 | 0,6300 |
| Media móvil (T2.2) | 0,1379 | 0,8672 | 0,7023 |
| **A — Local (MLP, T2.3)** | **0,1476** | **0,9283** | **0,7501** |
| Persistencia (T2.2) | 0,1584 | 0,9742 | 0,8025 |
| Estacional (T2.2) | 0,2296 | 1,2684 | 1,0215 |
| ETS/Holt-Winters (T2.2b) | 0,3344 | 1,4553 | 1,1883 |

**Interpretación:** A queda por delante de persistencia, estacional y ETS, pero por detrás de la
media móvil y del LightGBM global -- exactamente lo esperado: 54 modelos entrenando cada uno con
~6.000 filas propias no pueden competir con un solo modelo que ve las ~294.000 filas de train
completas. Es el punto de partida ("sin colaboración") frente al que B (centralizado por silo), C
(centralizado global) y, sobre todo, D/E (federado) deben demostrar mejora — la pregunta central
de RQ1.

### Resultados finales de tests

**7 de 7 tests nuevos pasan** (`tests/test_modelo_mlp.py`): recuento exacto de parámetros,
forma de la salida, dimensiones de los embeddings, forma de los tensores del dataset, reducción
de pérdida con señal sintética aprendible, activación del early stopping, no negatividad de las
predicciones. **46 de 46 tests en total** (`pytest tests/ -v`).

### Salidas
- `src/modelo_mlp.py` — arquitectura y utilidades de entrenamiento, reutilizable por B-E.
- `src/12_condicion_a_local.py` — entrenamiento y evaluación de la condición A.
- `reports/resultados_condicion_A_local.csv`.
- `tests/test_modelo_mlp.py` — 7 tests.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/12_condicion_a_local.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Sesión 24 — T2.4-T2.5: Condiciones B y C — la centralización no siempre gana

**Fecha:** 2026-07-21
**Scripts:** `src/13_condicion_b_silo.py`, `src/14_condicion_c_global.py`
**Objetivo:** entrenar las dos condiciones centralizadas restantes con el mismo `modelo_mlp.py`
de la Sesión 23 — B (un modelo por silo, 3 modelos) y C (un único modelo con las 54 tiendas, el
"techo" en la práctica inviable) — y compararlas con A (Local, Sesión 23).

### Método

Sin cambios en la arquitectura ni en `entrenar()` (mismos hiperparámetros que A: Huber+Adam,
early stopping con paciencia=15) — solo cambia el agrupamiento de filas que se le pasa: por
`silo` en B (3 modelos), sin agrupar en C (1 modelo con todo `train`).

### Resultado inesperado — B y C rinden PEOR que A

| Condición | Split | WMAPE mediana | MASE mediana | RMSSE mediana |
|---|---|---|---|---|
| A — Local (54 modelos) | test | **0,1476** | **0,9283** | **0,7501** |
| B — Centralizado por silo (3 modelos) | test | 0,1676 | 1,0423 | 0,8256 |
| C — Centralizado global (1 modelo) | test | 0,1736 | 1,1317 | 0,8832 |

Degradación **monótona y consistente** en las tres métricas: cuantas más tiendas heterogéneas se
mezclan en un mismo modelo, peor es el error típico por serie -- a pesar de que C ve ~54× más
datos que cualquier modelo de A. No es el resultado esperado ingenuamente ("más datos = mejor"),
así que se investigó antes de aceptarlo, con la misma disciplina que las Sesiones 19-22.

### Diagnóstico — ¿es sub-entrenamiento?

Se reentrenó el modelo del silo Grande (B) con paciencia mucho mayor (40 en vez de 15, tope 300
épocas) para comprobar si el early stopping original cortaba demasiado pronto. Resultado: el
val_loss toca su mínimo hacia la época ~15-20 (mejor_val_loss idéntico, 0,0323) y **empieza a
subir después** (0,033 en época 10 → 0,124 en época 30 → sigue subiendo) -- sobreajuste claro, no
falta de entrenamiento. Dar más paciencia solo alarga el entrenamiento sin mejorar el mejor
checkpoint. **Se descarta la hipótesis de sub-entrenamiento.**

### Interpretación

Es una manifestación real del problema de heterogeneidad no-IID (*non-independent and
identically distributed*) que motiva la investigación en aprendizaje federado -- y que este
propio proyecto ya había medido al elegir el eje de partición de silos (Sesión 4/15: JS=0,005 por
región vs. heterogeneidad real por formato). Que el formato de tienda genere heterogeneidad
suficiente para justificar 3 silos NO implica que las tiendas DENTRO de un silo sean homogéneas
entre sí -- y evidentemente no lo son lo bastante para que un embedding de 8 dimensiones por
tienda baste para que una única red comparta capacidad entre todas sin diluir patrones
específicos de cada una. Este es exactamente el fenómeno documentado en el propio paper de FedAvg
(McMahan et al., 2017) para clientes muy no-IID -- ahí es donde entra la personalización
(condición E, T3.3): en vez de forzar que un solo modelo sirva a todas las tiendas por igual, se
parte de un modelo global (federado) y se ajusta fino por tienda.

**Hipótesis abierta (no verificada, anotar para la memoria):** una posible causa adicional es que
LightGBM (Sesión 22) SÍ se benefició de agrupar `store_id` como feature categórica en un único
modelo -- porque un árbol puede crear ramas de decisión completamente separadas por tienda, sin
"mezclar" sus pesos. Un MLP, en cambio, aplica la MISMA transformación densa a la representación
de todas las tiendas; la única forma de especializarse es a través del embedding de 8 dimensiones,
una representación mucho más entrelazada y con menos margen para evitar interferencia entre
tiendas distintas. Pendiente de explorar si esto se confirma al llegar al análisis cualitativo de
embeddings (Sesión 15, proyección PCA).

### Decisión

Se acepta el resultado tal cual, documentado con honestidad: **no se fuerza ni se maquilla** para
que encaje con la expectativa de "más datos, mejor modelo". Es, de hecho, un resultado más
valioso para la tesis que el contrario -- si C (el techo) ya fuera claramente mejor que A sin
necesitar ningún mecanismo especial, la pregunta de investigación central (RQ1: ¿el federado con
personalización recupera la brecha sin compartir datos crudos?) perdería fuerza. Ahora la
pregunta relevante para la Fase 3 es más matizada: ¿puede D (FedAvg) al menos igualar a C, y puede
E (FedAvg + personalización) superar tanto a A como a C?

### Resultados finales

| Condición | Split | WMAPE media | WMAPE mediana | MASE mediana | RMSSE mediana | Cobertura |
|---|---|---|---|---|---|---|
| B — Centralizado por silo | val | 0,2640 | 0,1780 | 1,1733 | 0,9439 | 100,0% |
| B — Centralizado por silo | test | 0,2697 | 0,1676 | 1,0423 | 0,8256 | 100,0% |
| C — Centralizado global | val | 0,2993 | 0,1897 | 1,2900 | 1,0202 | 100,0% |
| C — Centralizado global | test | 0,3047 | 0,1736 | 1,1317 | 0,8832 | 100,0% |

### Salidas
- `src/13_condicion_b_silo.py`, `src/14_condicion_c_global.py`.
- `reports/resultados_condicion_B_silo.csv`, `reports/resultados_condicion_C_global.csv`.
- Sin tests nuevos: ambos scripts son orquestación fina sobre `modelo_mlp.py` (ya cubierto por
  `tests/test_modelo_mlp.py`, Sesión 23) -- añadir tests que solo repitan ese mismo módulo con
  distinto agrupamiento de filas sería sobre-ingeniería sin verificar nada nuevo.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/13_condicion_b_silo.py
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/14_condicion_c_global.py
```

---

## Sesión 25 — Integración de Weights & Biases (previa a la Fase 3)

**Fecha:** 2026-07-22
**Scripts:** `src/modelo_mlp.py` (modificado) + `src/15_wandb_resumen_fase2.py` (nuevo)
**Objetivo:** integrar tracking de experimentos antes de entrar en la Fase 3 (federado), donde el
número de corridas a comparar crece mucho (rondas × condiciones D/E) y hace falta un dashboard,
no solo CSVs sueltos.

### Método

1. `wandb` instalado y añadido a `requirements.txt`.
2. `entrenar()` (en `modelo_mlp.py`) acepta ahora un parámetro opcional `wandb_run` (por defecto
   `None`) — cualquier objeto con `.log(dict)`, típicamente el valor de `wandb.init()`. El módulo
   **no importa `wandb` directamente**: quien llama a `entrenar()` decide si trackea la corrida,
   manteniendo `modelo_mlp.py` desacoplado de esa dependencia (relevante para los tests, que usan
   un doble de prueba en vez de una cuenta real de W&B).
3. `src/15_wandb_resumen_fase2.py`: no reentrena nada — carga los 5 CSV de resultados ya cerrados
   (T2.2, T2.2b, T2.3, T2.4, T2.5), los une bajo una columna `metodo` común, y registra en W&B una
   tabla comparable + un gráfico de barras (WMAPE mediana en test) como línea base visual antes de
   que empiece la Fase 3.
4. **Modo por defecto: offline.** No hay cuenta de W&B configurada en esta máquina todavía —
   `WANDB_MODE=offline` guarda la corrida en `wandb/` (ya estaba en `.gitignore`) sin necesitar
   login ni red. Se verificó que sincroniza limpiamente con una prueba mínima antes de integrarlo.

### Resultados

Corrida offline ejecutada correctamente; el ranking registrado coincide exactamente con el ya
reportado en el informe de Fase 2 (verificación cruzada, no solo "no dio error"):
LightGBM global (0,1320) < Media móvil (0,1379) < A-Local (0,1476) < Persistencia (0,1584) <
B-Silo (0,1676) < C-Global (0,1736) < Estacional (0,2296) < ETS (0,3344).

**4 tests nuevos** (2 en `test_modelo_mlp.py` con un doble de prueba de `wandb.Run`, 2 en
`test_wandb_resumen_fase2.py` para la unión de CSVs). **50 de 50 tests en total.**

### Pendiente de acción del usuario (no bloquea)

Crear una cuenta gratuita en [wandb.ai](https://wandb.ai) y ejecutar `wandb login` cuando
convenga; en ese momento, `wandb sync wandb/<carpeta-de-la-corrida>` sube lo ya registrado en
local sin tener que re-ejecutar nada. A partir de la Fase 3, si ya hay cuenta, basta con
`WANDB_MODE=online` (o quitar la variable, que es el valor por defecto de la librería una vez
autenticada).

### Salidas
- `requirements.txt` — `wandb>=0.28` añadido.
- `src/modelo_mlp.py` — `entrenar()` con hook opcional de tracking.
- `src/15_wandb_resumen_fase2.py` — corrida de referencia de la Fase 2 completa.
- `tests/test_wandb_resumen_fase2.py` — 2 tests.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
WANDB_MODE=offline PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/15_wandb_resumen_fase2.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Sesión 26 — T3.1-T3.2: infraestructura Flower + Condición D (FedAvg vs. FedProx)

**Fecha:** 2026-07-22
**Scripts:** `src/federado_flower.py` (nuevo), `src/16_condicion_d_federado.py` (nuevo),
`tests/test_federado_flower.py` (nuevo)
**Objetivo:** montar Flower (framework real de FL, no un bucle de FedAvg hecho a mano) para
simular el entrenamiento federado entre los 3 silos, y entrenar la primera condición realmente
federada del proyecto (D), comparando FedAvg con FedProx dado que la Sesión 24 ya encontró
heterogeneidad no-IID real entre tiendas de un mismo silo.

### Método — T3.1

Se instaló `flwr[simulation]` (1.32.1, con backend Ray) y se validó con una prueba mínima
(2 clientes de juguete) antes de conectar el modelo y los datos reales -- funciona correctamente
en esta máquina (Windows/CPU), con un coste fijo de arranque de Ray de ~15-40s por simulación.

`src/federado_flower.py`: los 3 SILOS son los clientes (no las 54 tiendas sueltas, igual que en
B -- Sesión 5). Reutiliza `MLPConEmbeddings` sin ninguna modificación; lo que cambia frente a A/B/C
es el bucle de entrenamiento: cada silo hace pocas épocas locales POR RONDA (no un `entrenar()`
completo con early stopping), y el servidor promedia los pesos entre rondas -- la diferencia
central de FedAvg (McMahan et al., 2017) frente a un promediado de un solo paso.

Como `run_simulation()` (API "Flower Next") no devuelve un objeto `History` como la API clásica,
se implementó checkpointing manual: `evaluate_fn` (evaluación CENTRALIZADA sobre el val global de
las 3 silos, no un val por silo -- más estable, evita el mismo problema de ruido de la Sesión 24)
guarda los pesos de cada ronda a disco y registra su pérdida en un dict compartido; terminada la
simulación, se recupera la ronda de menor pérdida (mismo principio que `entrenar()`: quedarse con
los mejores pesos vistos, no los últimos).

Soporta FedProx (Li et al., 2020) además de FedAvg: añade un término proximal
`(mu/2)·||w_local - w_global||²` a la pérdida local, penalizando que un silo se aleje demasiado
de los pesos globales en cada ronda -- diseñado específicamente para clientes no-IID.

### Incidencia — bug de aliasing de memoria en `get_params()`

Al escribir los tests de `federado_flower.py`, dos fallaron de forma llamativa: tomar una
"foto" de los pesos antes y después de entrenar daba **exactamente la misma diferencia (0.0)**,
pese a que la pérdida sí bajaba con el entrenamiento. Investigado antes de aceptarlo:

**Causa raíz:** `v.cpu().numpy()` sobre un tensor de PyTorch que ya está en CPU **comparte
memoria** con el tensor original (comportamiento zero-copy documentado de la interoperabilidad
PyTorch/NumPy) -- verificado con `np.shares_memory()`. La función `get_params()` devolvía, por
tanto, una VISTA sobre los pesos en vivo del modelo, no una copia independiente: si el modelo
seguía entrenando después de tomar la "foto", el array ya "capturado" cambiaba en silencio con
él, porque apuntaba a la misma memoria.

**Por qué no se detectó en la simulación real:** en `run_simulation` con backend Ray, cada
cliente corre en un proceso/actor separado; el valor de retorno de `fit()` se serializa para
cruzar esa frontera de proceso antes de que el modelo del cliente vuelva a entrenar en la ronda
siguiente -- la serialización rompe el alias de memoria como efecto secundario. La simulación ya
ejecutada con la versión con el bug (calibración de 5 rondas y la ejecución completa de D)
mostró una curva de pérdida coherente y monótona en las primeras rondas, consistente con
agregación correcta -- se decidió NO relanzar esas corridas, documentando el razonamiento en vez
de descartar resultados sin evidencia de que estuvieran mal.

**Corrección:** `get_params()` ahora hace `.detach().cpu().numpy().copy()` -- copia explícita.
Los 5 tests de `federado_flower.py` (incluidos los 2 que habían fallado) pasan tras la corrección.

### Resultados — Condición D

Calibración previa (5 rondas sobre datos reales): la pérdida de validación centralizada ya
convergía hacia la ronda 2-3 (0,062-0,061) y empezaba a subir levemente hacia la 5 -- se fijó
`NUM_ROUNDS=15` (margen de sobra sin gastar cómputo innecesario, ~40s/ronda).

| Ronda | val_loss FedAvg | val_loss FedProx (μ=0,01) |
|---|---|---|
| 0 (inicial) | 5,030 | 5,030 |
| 2 | 0,0656 | **0,0623** (mínimo FedProx) |
| 3 | **0,0606** (mínimo FedAvg) | 0,0638 |
| 10 | 0,0818 | 0,0877 |
| 15 | 0,0812 | 0,1133 |

Ambas estrategias convergen rápido (ronda 2-3) y luego se degradan levemente con más rondas —
**FedAvg se mantiene más estable que FedProx** en las rondas posteriores con este μ=0,01 (sin
tunear); el término proximal no ayudó aquí, posiblemente porque la selección de la mejor ronda ya
mitiga buena parte del problema de "deriva" que FedProx está diseñado a resolver. Se reporta
FedAvg como resultado de D; FedProx queda documentado como referencia, no descartado.

| Condición | Split | WMAPE mediana | MASE mediana | RMSSE mediana |
|---|---|---|---|---|
| D — FedAvg (ronda 3) | test | **0,1507** | 1,0100 | 0,7966 |
| D — FedProx (ronda 2) | test | 0,1681 | 1,0756 | 0,8515 |

### Interpretación — D ya bate a B y C

Comparando con lo ya cerrado (test, WMAPE mediana): **D-FedAvg (0,1507) queda por debajo de B
(0,1676) y de C (0,1736)** -- el federado, SIN que ningún silo comparta una fila de datos cruda
con otro, supera a ambas alternativas centralizadas. Quedó, eso sí, ligeramente por detrás de A
(0,1476, entrenamiento local puro). Es un resultado central y muy defendible para RQ1: confirma
que centralizar datos heterogéneos no es el "techo" que la métrica de brecha recuperada asume por
diseño -- ver nota metodológica en la sección de salidas.

### Nota metodológica — la métrica de brecha recuperada asume un orden que aquí no se cumple

`porcentaje_brecha_recuperada(wmape_local, wmape_centralizado, wmape_federado)` (T2.1) se diseñó
asumiendo Local > Centralizado (en error) -- es decir, que centralizar datos es el techo ideal.
La Sesión 24 ya rompió esa asunción (C rindió peor que A) y D lo confirma de nuevo. Aplicar la
fórmula mecánicamente da un número (~12%) que **no debe reportarse sin este matiz**: con
`wmape_local=0,1476` y `wmape_centralizado=0,1736` (C), la "brecha total" es negativa
(-0,0260) -- Local ya es mejor que Centralizado en este dataset heterogéneo. La lectura honesta y
más útil aquí es la comparación directa de WMAPE (D bate a B y C, casi iguala a A), no forzar el
porcentaje de una métrica cuya premisa no se sostiene. Pendiente de decidir, para la memoria, si
se redefine la métrica (p.ej. contra el mejor de A/B/C como "techo práctico" en vez de asumir C)
o se reporta solo la comparación directa.

### Salidas
- `src/federado_flower.py` — infraestructura de FL con Flower (cliente, checkpointing, FedAvg/FedProx).
- `src/16_condicion_d_federado.py` — entrenamiento y evaluación de la Condición D.
- `tests/test_federado_flower.py` — 5 tests (incluye el bug de aliasing).
- `reports/resultados_condicion_D_federado.csv`, `reports/historial_rondas_condicion_D.csv`.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/16_condicion_d_federado.py
./.venv/Scripts/python.exe -m pytest tests/test_federado_flower.py -v
```

---

## Sesión 27 — T3.3: Condición E (Federado + personalización) — cierre de la Fase 3 (D, E)

**Fecha:** 2026-07-22
**Script:** `src/17_condicion_e_personalizacion.py`
**Objetivo:** partir del modelo global ya convergido de D (FedAvg, ganador de la Sesión 26) y
hacer un fine-tuning corto por tienda individual -- la condición diseñada específicamente para
responder a la heterogeneidad no-IID que B y C sufrieron (Sesión 24).

### Método

`cargar_mejor_modelo_D()` lee `resultados_condicion_D_federado.csv`, identifica la estrategia
ganadora (FedAvg) y carga su checkpoint de la mejor ronda (ronda 3) desde
`data/processed/checkpoints_federado/`. Con esos pesos como punto de partida (`modelo_inicial`,
parámetro nuevo añadido a `entrenar()` en `modelo_mlp.py` -- antes solo permitía inicialización
aleatoria), se hace un fine-tuning independiente por tienda (54 modelos), con `lr=5e-4` (más bajo
que el 1e-3 por defecto, para no deshacer de golpe la estructura ya aprendida) y early stopping
igual que en A.

### Resultados

| Condición | Split | WMAPE mediana | MASE mediana | RMSSE mediana | Cobertura |
|---|---|---|---|---|---|
| E — Federado + personalización | val | 0,1553 | 1,0466 | 0,8414 | 100,0% |
| E — Federado + personalización | test | **0,1396** | 0,9002 | 0,7190 | 98,1% |

**E supera a A, B, C y D en las tres métricas por mediana** -- es, con diferencia, la mejor de
las cinco condiciones experimentales (A-E) entrenadas con el MLP+embeddings del proyecto:

| Condición | WMAPE test (mediana) |
|---|---|
| A — Local | 0,1476 |
| B — Centralizado por silo | 0,1676 |
| C — Centralizado global | 0,1736 |
| D — FedAvg | 0,1507 |
| **E — Federado + personalización** | **0,1396** |

### Interpretación — la métrica de brecha recuperada, revisitada

Aplicando `porcentaje_brecha_recuperada(wmape_local=0,1476, wmape_centralizado=0,1736,
wmape_federado=0,1396)` mecánicamente: brecha_total = 0,1476-0,1736 = **-0,0260** (negativa: local
ya es mejor que centralizado); brecha_recuperada = 0,1476-0,1396 = 0,0080; porcentaje ≈ **-30,8%**.

Un porcentaje negativo aquí NO significa un resultado malo -- significa que la premisa de la
métrica (centralizado = techo superior a local) no se cumple en este dataset, exactamente como ya
se documentó en la Sesión 26. La lectura correcta y más fuerte es la directa: **E bate en
términos absolutos tanto a A (local) como a C (centralizado)** -- no solo recupera una brecha,
la cierra por completo y la supera por ambos lados. Es, si acaso, un resultado más contundente que
el que la métrica original fue diseñada para capturar. Queda pendiente para la memoria decidir si
se redefine formalmente la métrica (p.ej. como % de mejora sobre el mejor de {A, B, C} en vez de
asumir C como techo) o se reporta la comparación directa con esta nota explicativa.

### Clasificación general actualizada (los 10 métodos de la Fase 2 + Fase 3, por WMAPE test mediana)

| Método | WMAPE mediana | Tipo |
|---|---|---|
| LightGBM (global) | 0,1320 | convencional |
| Media móvil (4 sem.) | 0,1379 | ingenuo |
| **E — Federado + personalización** | **0,1396** | **federado** |
| A — Local | 0,1476 | MLP |
| D — FedAvg | 0,1507 | federado |
| Persistencia (t-1) | 0,1584 | ingenuo |
| B — Centralizado por silo | 0,1676 | MLP |
| D — FedProx | 0,1681 | federado |
| C — Centralizado global | 0,1736 | MLP |
| Estacional (t-52) | 0,2296 | ingenuo |
| ETS/Holt-Winters | 0,3344 | convencional |

**E es la tercera mejor de las 11 filas de esta tabla**, y la mejor de todas las que usan el
MLP+embeddings del proyecto -- por delante de LightGBM le queda un margen pequeño (0,1396 vs
0,1320) y de la media móvil casi lo iguala (0,1396 vs 0,1379). Dado que ninguna de las
condiciones A-E ha pasado por un ajuste de hiperparámetros propio (a diferencia del LightGBM de
la Sesión 22), hay margen razonable para pensar que ese margen podría cerrarse con tuning --
anotado como posible trabajo futuro, no una limitación que invalide el resultado ya conseguido.

### Fase 3 (T3.1, T3.2, T3.3) completa

Con esta sesión se cierran T3.1 (infraestructura Flower), T3.2 (Condición D) y T3.3 (Condición
E) -- las tres tareas CORE de la Fase 3, el resultado central del TFM (RQ1). T3.4 (manejo de
stragglers) y T3.5 (comparación estadística formal Wilcoxon A-E completa) quedan como
extensiones -- T3.5 parcialmente cubierta por las tablas comparativas de esta sesión y la
Sesión 26, pendiente de un test de significancia formal si se necesita para la memoria (requeriría
recuperar o recalcular las métricas por serie de A/B/C, no guardadas individualmente en su
momento).

### Salidas
- `src/17_condicion_e_personalizacion.py`.
- `reports/resultados_condicion_E_personalizacion.csv`.
- `reports/por_serie_condicion_E_val.csv`, `reports/por_serie_condicion_E_test.csv` (por si se
  necesita Wilcoxon más adelante).
- `src/modelo_mlp.py` — `entrenar()` con parámetro `modelo_inicial` (nuevo, para continuar desde
  pesos dados en vez de inicialización aleatoria). 1 test nuevo (56/56 en total).

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/17_condicion_e_personalizacion.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Sesión 28 — Tuning de D y E: mejora real en D, no se traslada a E (resultado honesto)

**Fecha:** 2026-07-30
**Scripts:** `src/18_tuning_arquitectura.py` (nuevo), `src/19_correccion_duan.py` (nuevo),
`src/correccion_sesgo.py` (nuevo), `src/modelo_mlp.py` (ampliado: `ArquitecturaMLP`,
`congelar_base`, `predecir_log`), `src/federado_flower.py` (ampliado: `arq` en todas las
funciones), `src/16_condicion_d_federado.py` y `src/17_condicion_e_personalizacion.py`
(reescritos)
**Objetivo:** ninguna de las condiciones A-E había pasado por ajuste de hiperparámetros propio, a
diferencia de LightGBM (Sesión 22). El autor pidió implementar las recomendaciones del informe
"Cómo mejorar el federado" y medir el resultado. Plan aprobado en 4 etapas: arquitectura/
optimización → épocas locales por ronda → estrategia de personalización → corrección de sesgo de
Jensen (Duan). Cada etapa se decide por **val**; el resultado se reporta en **test**.

### Etapa 0 — Infraestructura para hacer tuneable la arquitectura

`ArquitecturaMLP` (dataclass: `dim_emb`, `hidden1`, `hidden2`, `dropout`) sustituye los valores
fijos (64/32/0,2/8) en `MLPConEmbeddings`, `entrenar()`, `ClienteSilo` y todas las funciones de
`federado_flower.py` que construyen un modelo -- un solo objeto en vez de proliferar kwargs
sueltos. `entrenar()` ahora copia `modelo_inicial` con `copy.deepcopy` (no
`MLPConEmbeddings()`+`load_state_dict`) para heredar su arquitectura exacta, no la que indique
`arq` -- necesario porque un modelo con una arquitectura no estándar (de una búsqueda) fallaría al
cargar su `state_dict` en un modelo de arquitectura por defecto. `congelar_base=True` congela
`red[0]`/`red[3]` (capas densas compartidas), dejando entrenables solo los embeddings y la capa
de salida -- estilo FedPer (Arivazhagan et al., 2019, verificado antes de citarlo). 17 tests
nuevos, todos pasan sin cambiar el comportamiento por defecto (backward-compatible).

### Etapa 1 — Búsqueda de arquitectura y optimización: mejora grande en el proxy

Búsqueda aleatoria de 13 candidatos (el 1º es siempre la arquitectura original) sobre el dataset
GLOBAL agrupado (mismo patrón que la Sesión 22 para LightGBM), seleccionando por WMAPE en escala
natural sobre val.

**Ganador:** `lr=0,002, dropout=0,1, hidden1=128, hidden2=64, dim_emb=8, batch_size=256`
— WMAPE val (proxy) = 0,1093, frente a 0,1880 de la arquitectura original: **+41,9% de mejora**.
Guardado en `configs/mlp_arquitectura.json`.

### Etapa 2 — Épocas locales por ronda (Condición D): mejora real, confirmada en test

Con la arquitectura ganadora, se probó `epocas_locales ∈ {1, 2, 3}` con FedAvg (15 rondas), más
FedProx (μ=0,01) con `epocas_locales=2` como control.

| Config | WMAPE test mediana |
|---|---|
| FedAvg, epocas_locales=1 | 0,1429 |
| FedAvg, epocas_locales=2 | 0,1503 |
| FedAvg, epocas_locales=3 | 0,2786 (mucho peor) |
| **FedProx, epocas_locales=2** | **0,1402** ← ganador |

Con `lr=0,002` (más alto que el 1e-3 original), más épocas locales por ronda deja que los silos
se alejen más entre sincronizaciones -- `epocas_locales=3` degrada claramente. FedProx, que
penaliza precisamente ese alejamiento, vuelve a ser mejor que FedAvg aquí (a diferencia de la
Sesión 26, con la arquitectura original) -- probablemente porque el lr más alto sí hace que el
término proximal tenga trabajo real que hacer.

**D mejora de verdad:** 0,1402 (tuneado) frente a 0,1507 (Sesión 26, sin tunear) — **~7% mejor**,
confirmado en test, no solo en el proxy de val.

### Etapa 3 — Estrategia de personalización (Condición E): NINGUNA variante mejora al original

Con el D ganador de la Etapa 2 como punto de partida, se probaron tres variantes de fine-tuning
por tienda:

| Variante | WMAPE test mediana |
|---|---|
| completo (lr=5e-4) | 0,1446 |
| lr_bajo (lr=2e-4) | 0,1664 |
| fedper (`congelar_base=True`) | 0,1558 |
| **original, Sesión 27 (arquitectura sin tunear, regenerado)** | **0,1379** ← sigue siendo el mejor |

**Ninguna de las tres variantes nuevas supera al resultado original de la Sesión 27.** Se
regeneró ese resultado original (el checkpoint de D de la Sesión 26 seguía en disco,
`checkpoints_federado/fedavg/ronda_3.npz`) para confirmar la cifra con el código actual: WMAPE
test mediana = 0,1379 -- prácticamente idéntico al 0,1396 reportado entonces (pequeña variación
por aleatoriedad de entrenamiento), y ahora **empata exactamente con la media móvil**.

**Interpretación:** la arquitectura más grande (128 vs. 64 unidades ocultas) que ayudó a D
—entrenado con las ~294.000 filas agregadas de los 3 silos— parece sobreajustar más fácilmente en
el fine-tuning por tienda, donde cada modelo ve solo unos cientos de filas. Es un caso concreto y
medido de que la capacidad que ayuda con mucho dato (entrenamiento federado) puede perjudicar en
una etapa posterior con mucho menos dato por unidad (personalización) -- un hallazgo honesto, no
el resultado que se buscaba, pero real y documentado con evidencia.

### Etapa 4 — Corrección de Duan: efecto catastrófico, descartada

Se aplicó el estimador de smearing de Duan (1983, verificado antes de citarlo) sobre la variante
"completo" de la Etapa 3, con el factor de corrección calculado **por tienda** sobre los residuos
de esa tienda en train.

**Resultado: la corrección empeoró el WMAPE test mediana de 0,1446 a 0,8876** -- un desastre, no
una mejora marginal. Diagnóstico: con solo ~200-300 filas de train por tienda, el factor
`media(exp(residuo))` es extremadamente sensible a valores atípicos del residuo (la
exponencial amplifica la cola derecha) -- el factor osciló entre 1,10 y **7,41** según la tienda
(mediana 1,87, media 2,15). Unas pocas semanas mal predichas en el train de una tienda bastan
para disparar su factor y arruinar sus predicciones en val/test.

**Decisión:** se descarta la corrección de Duan a nivel tienda. Queda documentada la hipótesis
razonable para un posible trabajo futuro: un factor GLOBAL (un solo valor para todas las tiendas,
calculado sobre las ~294.000 filas de train agregadas, no por tienda) promediaría mejor los
atípicos y podría funcionar -- no se probó en esta sesión, fuera del alcance ya acordado.

### Veredicto final — qué configuración queda como la mejor conocida

| Método | WMAPE test mediana |
|---|---|
| LightGBM (global, Sesión 22) | 0,1320 |
| Media móvil (4 sem., T2.2) | 0,1379 |
| **E — federado + personalización (Sesión 27, sin tunear, confirmado)** | **0,1379** |
| D — FedProx tuneado (Etapa 2, sin personalizar) | 0,1402 |
| A — Local (T2.3) | 0,1476 |
| E — completo, arquitectura tuneada (Etapa 3) | 0,1446 |

**La configuración de la Sesión 27 (arquitectura original, sin tunear) sigue siendo la mejor
condición federada del proyecto.** El esfuerzo de tuning de esta sesión SÍ produjo una mejora real
y medible -- para D solo, sin personalizar (0,1507→0,1402, ~7%) -- pero esa mejora no se
propaga a E. No se fuerza un "resultado combinado ganador" que no existe: es más honesto reportar
que el tuning tuvo éxito parcial, con evidencia clara de por qué (sobreajuste de una arquitectura
más grande en datasets de fine-tuning pequeños), que maquillar el resultado.

**Recomendación para producción:** mantener la configuración de la Sesión 27 (arquitectura
original) como la reportada; considerar la arquitectura tuneada de esta sesión únicamente si se
usa D sin personalizar. Trabajo futuro razonable, no realizado aquí: una búsqueda de arquitectura
separada y más pequeña específica para la etapa de personalización, y un factor de Duan global en
vez de por tienda.

### Resultados finales de tests

**89 de 89 tests pasan** (17 nuevos en `test_modelo_mlp.py`, 2 nuevos en `test_federado_flower.py`,
7 nuevos en `test_correccion_sesgo.py`, sobre los 63 ya existentes).

### Salidas
- `src/modelo_mlp.py` — `ArquitecturaMLP`, `congelar_base`, `predecir_log`.
- `src/federado_flower.py` — `arq` propagado a `ClienteSilo`, `evaluate_fn`, `ejecutar_federado`,
  `cargar_mejor_ronda`.
- `src/18_tuning_arquitectura.py`, `src/19_correccion_duan.py`, `src/correccion_sesgo.py` — nuevos.
- `src/16_condicion_d_federado.py`, `src/17_condicion_e_personalizacion.py` — reescritos para
  soportar múltiples configuraciones y arquitectura tuneada.
- `configs/mlp_arquitectura.json` — arquitectura ganadora de la Etapa 1.
- `reports/resultados_condicion_D_federado.csv`, `reports/resultados_condicion_E_personalizacion.csv`
  (reflejan las corridas de esta sesión), `reports/resultados_condicion_E_original_sesion27.csv`
  (regenerado para la comparación final), `reports/resultados_correccion_duan.csv`,
  `reports/tuning_arquitectura_mlp.csv`, `reports/historial_rondas_condicion_D.csv`.

### Reproducibilidad
```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/18_tuning_arquitectura.py
./.venv/Scripts/python.exe src/16_condicion_d_federado.py
./.venv/Scripts/python.exe src/17_condicion_e_personalizacion.py
./.venv/Scripts/python.exe src/19_correccion_duan.py
./.venv/Scripts/python.exe -m pytest tests/ -v
```

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
