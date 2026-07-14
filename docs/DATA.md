# DATA.md — Diccionario de datos y plan de uso

> Referencia permanente de qué contienen los datos, su granularidad exacta, y cómo se usa cada
> pieza a lo largo del proyecto. Todas las cifras de este documento están **verificadas
> directamente sobre los ficheros** (no de memoria) — ver comandos de reproducibilidad al final.
> Complementa a `docs/RESEARCH_LOG.md` (que registra el proceso) — este documento es el
> **estado actual y estable** de qué son los datos.

---

## 1. Resumen de una frase por archivo

| Archivo | Qué es | Filas | Granularidad |
|---|---|---|---|
| `train.csv` | Ventas — el corazón del dataset | 3.000.888 | tienda × familia × **día** |
| `stores.csv` | Metadatos de cada tienda | 54 | 1 fila por tienda |
| `transactions.csv` | Nº de tickets de caja por tienda y día | 83.488 | tienda × día (con huecos) |
| `oil.csv` | Precio diario del petróleo WTI (macro) | 1.218 | día (bursátil) |
| `holidays_events.csv` | Calendario de festivos y eventos de Ecuador | 350 | evento (no 1 por día) |
| `test.csv` / `sample_submission.csv` | Formato de la competición Kaggle | 28.512 | **no se usa** en este TFM |

---

## 2. Cada archivo, en detalle

### 2.1 `train.csv` — el archivo principal

```
id, date, store_nbr, family, sales, onpromotion
```

- **Rango temporal:** 2013-01-01 → 2017-08-15 (**1.684 días**, ~4 años y 8 meses).
- **54 tiendas × 33 familias de producto** — el panel está **completo**: 54×33×1.684 = 3.000.888 filas exactas, ni una de más ni de menos. Toda combinación tienda-familia-día tiene una fila, aunque la venta sea 0.
- **`sales`** (float): unidades vendidas (no € — hay productos que se venden por peso, de ahí que sea decimal). Rango 0 – 124.717. **Sin negativos.** El **31,3% de las filas tienen venta = 0** — ojo, esto mezcla dos situaciones distintas (ver gotcha 2.6.1).
- **`onpromotion`** (int): nº de productos de esa familia en promoción ese día en esa tienda. Rango 0–741. Es la única señal de política comercial disponible (no hay precio, ver `RESEARCH_LOG.md` Sesión 5).
- **`family`**: 33 categorías (GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY...). Es la unidad de producto — no hay SKU/item individual en esta versión del dataset.

### 2.2 `stores.csv` — metadatos de tienda

```
store_nbr, city, state, type, cluster
```

- 54 filas, una por tienda, **sin nulos**.
- **`city`**: 22 ciudades. **`state`**: 16 provincias.
- **`type`**: 5 valores (A-E) — clasificación **propia de Favorita**, de significado no revelado (probablemente formato/tamaño/ubicación mezclados). Es la base de nuestra partición en silos.
- **`cluster`**: 17 valores — agrupación de similitud, también interna de Favorita, no usada como criterio de silo (se descartó en la Sesión 4 por ser menos defendible narrativamente).

### 2.3 `transactions.csv` — tráfico de clientes

```
date, store_nbr, transactions
```

- **83.488 filas**, pero el panel **NO está completo**: se esperarían 90.828 (54 tiendas × 1.682 días con dato) y faltan **7.340** — probablemente días en que la tienda estuvo cerrada.
- `transactions`: rango 5 – 8.359 tickets/día.
- **Uso decidido en esta sesión:** validar (junto con `sales`) el criterio de partición en silos — correlación 0,91 con venta total por tienda (ver `RESEARCH_LOG.md` Sesión 8). No se usa como feature de entrenamiento por ahora (sería filtrar información casi redundante con el propio target).

### 2.4 `oil.csv` — precio del petróleo (variable macro)

```
date, dcoilwtico
```

- 1.218 filas, 2013-01-01 → 2017-08-31. **43 valores nulos** en `dcoilwtico` (días sin cotización — fines de semana/festivos bursátiles en EEUU, no de Ecuador).
- Cobertura parcial respecto a los 1.684 días de `train.csv` porque el petróleo solo cotiza en días hábiles.
- **Gotcha de uso:** al agregar a semana (T1.2), la media semanal absorbe los huecos de forma natural — no hace falta interpolar día a día.

### 2.5 `holidays_events.csv` — calendario de eventos

```
date, type, locale, locale_name, description, transferred
```

- 350 filas, 2012-03-02 → 2017-12-26 (rango **más amplio** que `train.csv` — hay eventos de 2012 y de finales de 2017 fuera de nuestro rango de ventas, se filtran al usar).
- **`type`**: Holiday (221) · Event (56) · Additional (51) · Transfer (12) · Bridge (5) · Work Day (5).
- **`locale`**: National (174) · Local (152) · Regional (24) — determina a qué tiendas afecta (Local/Regional solo a las de esa ciudad/provincia, vía `stores.city`/`state`).
- **`transferred`**: 12 festivos que se trasladaron de fecha (el festivo "de verdad" cae en la fecha del registro `type=Transfer` correspondiente, no en la fecha original marcada `transferred=True`) — **hay que cruzar ambos registros**, no tratarlos de forma independiente.
- **38 fechas con más de un evento simultáneo** (p. ej. un festivo nacional y uno local coinciden) — la feature de "festivo" debe poder acumular varios flags por semana, no asumir uno solo.

### 2.6 Gotchas encontrados en esta revisión (nuevos, no documentados antes)

**2.6.1 — Ocho tiendas abrieron durante el periodo del dataset** (no las 54 desde el día 1):

| Tienda | Primera venta registrada |
|---|---|
| 36 | 2013-05-09 |
| 53 | 2014-05-29 |
| 20 | 2015-02-13 |
| 29 | 2015-03-20 |
| 21 | 2015-07-24 |
| 42 | 2015-08-21 |
| 22 | 2015-10-09 |
| **52** | **2017-04-20** (¡solo 4 meses de historia real!) |

**Impacto:** las filas anteriores a la apertura real tienen `sales=0`, pero ese cero **no significa "no hubo demanda"**, significa "la tienda no existía". Si no se trata, contamina lags/medias móviles con ceros artificiales y sesga el entrenamiento de esas tiendas — especialmente grave para la tienda 52, con menos de un año de datos utilizables. **Se añade como tarea a T1.2/T1.4** (recortar el histórico de cada tienda a partir de su apertura real, y vigilar T3 si la 52 acaba en un silo con muy poca historia).

**2.6.2 — Ya documentado en la Sesión 3:** terremoto de Ecuador de abril 2016 (pico de ventas atípico) — se maneja con Huber loss (Sesión 5).

---

## 3. Granularidad: la cruda vs. la de modelado

| | Granularidad **cruda** (tal cual llega) | Granularidad **de modelado** (la que usamos) |
|---|---|---|
| Tiempo | día | **semana** (suma de ventas, media de promo/transacciones) |
| Producto | familia (33) | familia (sin cambios — ya es la unidad más fina disponible) |
| Tienda | tienda individual (54) | tienda individual, agrupada en 3 **silos** (Grande/Mediano/Pequeño) para el federado |

**Por qué semana y no día:** ya justificado en `RESEARCH_LOG.md` Sesión 5 — reduce ruido, alinea con el ciclo de reposición real de un supermercado, y hace manejable el nº de rondas de entrenamiento.

---

## 4. Plan de uso de principio a fin (mapeado a las fases del plan)

| Fase / Tarea | Qué archivo(s) | Cómo se usan |
|---|---|---|
| **T1.1** (silos) | `stores.csv` + `train.csv` (agregado) + `transactions.csv` | `type` de tienda + venta/tráfico → asignación de silo (Grande/Mediano/Pequeño) |
| **T1.2** (dataset de modelado) | `train.csv`, `oil.csv`, `holidays_events.csv` | Agregación a tienda×familia×semana; join de petróleo (media semanal) y festivos (flags por semana y locale); recorte de historial por tienda según fecha real de apertura (gotcha 2.6.1) |
| **T1.3** (corte temporal) | dataset ya agregado | Walk-forward: últimas ~8 semanas = test, 8 previas = val, resto = train — **por silo** |
| **T1.4** (features) | dataset agregado | Lags (1/2/4/8 sem.), medias móviles, `onpromotion` semanal, calendario, día de pago, embeddings de familia/tienda |
| **T2.2b** (baselines convencionales) | dataset agregado | ETS/Holt-Winters y LightGBM entrenados **sobre el mismo dataset y split** que el resto, para comparación justa (RQ2) |
| **T2.3-T2.5** (A/B/C) | dataset agregado, particionado por silo | Local = solo su tienda; Centralizado-silo = todas las tiendas de un silo; Centralizado-global = las 3 |
| **T3** (federado) | dataset agregado, particionado por silo | Cada silo entrena en local con **sus propias filas**; solo se comparten pesos (nunca estas tablas) — mecánica descrita en `RESEARCH_LOG.md` Sesión 5 |
| **T4b.2** (valor de negocio) | resultados de T3 (WMAPE) | Traducir la mejora de precisión a € — no requiere volver a tocar los datos crudos |
| **`transactions.csv`** | — | Solo se usa en T1.1 (validación del criterio de silo). No entra como feature de entrenamiento (sería casi redundante con el propio target). |
| `test.csv` / `sample_submission.csv` | — | **No se usan** — nuestro propio corte walk-forward reemplaza el split de la competición Kaggle. |

---

## 5. Lo que NO tenemos (limitaciones a citar en la memoria)

- **Precio** — ni de lista ni pagado (Sesión 5).
- **SKU/item individual** — solo familia agregada (33 categorías).
- **Superficie de tienda (m²) ni nº de referencias** — por eso el criterio de "formato" se infiere de venta/tráfico, no de atributos de diseño (Sesión 8).
- **Datos de clientes/fidelización** — no hay nada a nivel de cliente, todo es agregado por tienda.

---

## 6. Reproducibilidad de las cifras de este documento

```bash
cd "C:\Users\alefl\OneDrive\Escritorio\tfm-forecasting-federado"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -c "import pandas as pd; ..."
# Ver el comando completo ejecutado en RESEARCH_LOG.md, Sesión 8.
```
