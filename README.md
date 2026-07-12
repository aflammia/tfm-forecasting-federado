# Forecasting federado de demanda en supermercados

Trabajo de Fin de Máster · Máster en Análisis de Datos en Ingeniería (Tecnun)

## Idea

¿Puede una red de supermercados independientes predecir su demanda **entrenando juntos, sin compartir sus datos de venta**? Este proyecto compara, con métricas y rigor estadístico, tres formas de construir un modelo de previsión de demanda:

- **Local** — cada tienda entrena sola (poca información).
- **Centralizado** — se juntan todos los datos (mejor precisión, pero inviable entre negocios independientes).
- **Federado** — se entrena en conjunto sin mover los datos crudos (la propuesta).

**Hipótesis:** `Local < Federado ≈ Centralizado`, y la personalización por tienda mejora los casos más atípicos.

## Datos

[Corporación Favorita](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) — cadena real de supermercados de Ecuador: 54 tiendas en 22 ciudades, ventas diarias por familia de producto (33 familias), 2013-2017. Se particiona por **formato de tienda** (grande / mediano / pequeño ≈ hiper / súper / proximidad) en 3 silos que actúan como supermercados independientes que no comparten datos. El formato es el reparto con mayor heterogeneidad de comportamiento de compra entre silos (ver `src/02_silo_strategy.py`).

## Estado

🚧 En construcción — fase de datos y análisis exploratorio.

## Estructura

```
src/          código (datos, modelos, federado, evaluación)
data/         datos crudos y procesados (no versionados)
configs/      configuración de experimentos
notebooks/    exploración
reports/      figuras y resultados
```
