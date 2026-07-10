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

Se generó un token de API de Kaggle (formato nuevo `KGAT_...`) y se guardó en `C:\Users\alefl\.kaggle\access_token`. **Nota de seguridad:** el token fue compartido inicialmente por el usuario en el chat de la sesión; queda registrado aquí que se marcó como pendiente su regeneración (ver `STATE.md`, sección "Pendiente"), práctica estándar cuando un secreto ha quedado expuesto en un canal no cifrado de forma persistente.

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
Requisito previo (documentado para reproducibilidad): el usuario debe haber aceptado las reglas de la competición en `kaggle.com/competitions/store-sales-time-series-forecasting/rules` — sin este paso, la API devuelve error 403.

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

`test.csv` y `sample_submission.csv` pertenecen al formato original de la competición Kaggle (para su leaderboard) y **no se usan** en este TFM — nuestro propio corte de validación temporal (walk-forward) se construye a partir de `train.csv` en la Fase 1 del plan, ya que necesitamos control total sobre las fechas de corte para la comparación entre condiciones experimentales.

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
