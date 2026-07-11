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
