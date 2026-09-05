# REFERENCIAS.md — Bibliografía en formato APA (7.ª edición)

> Recopilación viva de todas las fuentes externas (papers, datasets, documentación técnica) en las
> que se apoyan las decisiones del proyecto. Se añade una entrada **en el momento en que se cita**
> una fuente nueva en `RESEARCH_LOG.md` — nunca al final, para no perder la trazabilidad de qué
> apoyó qué decisión. Formato: APA 7.ª edición. Este archivo se copia directamente a la sección de
> referencias de la memoria.
>
> **Convención de cada entrada:** cita APA completa · para qué se usó · en qué sesión del
> `RESEARCH_LOG.md` se citó por primera vez.

---

## Fundamentos de Federated Learning

**McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).** Communication-efficient
learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on
Artificial Intelligence and Statistics (AISTATS)*, 1273–1282. https://proceedings.mlr.press/v54/mcmahan17a.html

> Paper original del algoritmo FedAvg — fórmula de agregación ponderada, arquitectura cliente-servidor,
> justificación de por qué se sincroniza en rondas cortas y no una vez al final. Base de toda la Fase 3
> del proyecto. Citado por primera vez: Sesión 5 (diseño del algoritmo).

**Long, Y., Xu, L., Zheng, G., & Brintrup, A. (2025).** PA-CFL: Privacy-adaptive clustered federated
learning for transformer-based sales forecasting on heterogeneous retail data. *arXiv*.
https://arxiv.org/abs/2503.12220

> Paper de referencia más cercano a este TFM: FL aplicado a forecasting de ventas retail sobre datos
> heterogéneos. Evidencia clave de que particionar un único dataset real por un eje que genere
> heterogeneidad (en su caso, región geográfica) es la metodología estándar del campo — no existe
> ningún estudio que use cadenas competidoras reales como silos, porque ese dato no es público.
> Citado por primera vez: Sesión 15 (estado del arte de la partición en silos).

**Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).** Federated
optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems*, 2, 429–450.
https://arxiv.org/abs/1812.06127

> Paper original de FedProx — añade un término proximal (μ/2·‖w_local − w_global‖²) a la pérdida
> local, diseñado específicamente para clientes no-IID (systems + statistical heterogeneity).
> Usado como estrategia alternativa a FedAvg en la Condición D, dado que la Sesión 24 ya había
> encontrado heterogeneidad no-IID real entre tiendas de un mismo silo. Citado por primera vez:
> Sesión 26 (Condición D — FedAvg vs. FedProx).

**Arivazhagan, M. G., Aggarwal, V., Singh, A. K., & Choudhary, S. (2019).** Federated learning
with personalization layers. *arXiv*. https://arxiv.org/abs/1912.00818

> Paper original de FedPer — combina una "base" compartida entrenada de forma federada con capas
> de personalización que se entrenan solo localmente por cliente, sin re-entrenar la base. Base
> conceptual de la variante `congelar_base` de la Condición E (congelar las capas densas
> compartidas y afinar solo embeddings + capa de salida por tienda). Citado por primera vez:
> Sesión 28 (tuning de la personalización).

---

## Corrección de sesgo en transformaciones logarítmicas

**Duan, N. (1983).** Smearing estimate: A nonparametric retransformation method. *Journal of the
American Statistical Association*, 78(383), 605–610. https://doi.org/10.1080/01621459.1983.10478017

> Un modelo que minimiza error en escala log y deshace la transformación con `exp(ŷ)−1` subestima
> sistemáticamente la media en escala natural (desigualdad de Jensen). El "smearing estimate" de
> Duan corrige este sesgo multiplicando la predicción retransformada por la media de
> `exp(residuo_log)` sobre train, sin asumir una distribución paramétrica del residuo. Aplicado
> como corrección post-hoc sobre las predicciones ya generadas (Etapa 4 del tuning). Citado por
> primera vez: Sesión 28.

---

## Métodos de forecasting en retail (baselines convencionales, T2.2b)

**Petropoulos, F., Grushka-Cockayne, Y., Siemsen, E., & Spiliotis, E. (2024).** Wielding Occam's razor:
Fast and frugal retail forecasting. *Journal of the Operational Research Society*.
https://doi.org/10.1080/01605682.2024.2421339

> Evidencia de que LightGBM es el método de referencia en las competiciones de forecasting de retail más
> recientes (incluida M5), y contraste con los métodos estadísticos clásicos (ARIMA, Holt-Winters/ETS).
> Base de la decisión de usar LightGBM + ETS/Holt-Winters como baselines convencionales, descartando
> ARIMA y Prophet. Citado por primera vez: Sesión 7/9 (reenfoque técnico+negocio, decisión de baselines).

---

## Data clean rooms (capítulo de negocio, RQ4)

**Sherpa.ai. (s.f.).** Federated learning vs data clean rooms. Sherpa.ai Blog.
https://sherpa.ai/blog/federated-learning-vs-data-clean-rooms-2/ (Recuperado el 14 de julio de 2026)

**Snowflake. (s.f.).** Data clean rooms. Snowflake Inc.
https://www.snowflake.com/en/product/features/data-clean-rooms/ (Recuperado el 14 de julio de 2026)

**Digital Applied. (2026).** Data clean rooms 2026: A marketer's decision guide. Digital Applied Blog.
https://www.digitalapplied.com/blog/data-clean-rooms-advertising-2026-marketer-decision-guide
(Recuperado el 14 de julio de 2026)

> Las tres fuentes documentan que los data clean rooms operan sobre datos que sí se mueven a un entorno
> compartido (aunque el output sea agregado) y están diseñados para responder preguntas sobre datos
> existentes, no para entrenar modelos predictivos de forma nativa — la base del argumento de la Fase 4b
> (por qué el federated learning es la herramienta correcta para esta tarea, no un sustituto universal
> de los clean rooms). Citadas por primera vez: Sesión 7 (reenfoque técnico+negocio).

---

## Privacidad y marco regulatorio (RQ4)

**Agencia Española de Protección de Datos, & Supervisor Europeo de Protección de Datos. (2025).**
TechDispatch: Aprendizaje federado (n.º 1). AEPD y EDPS.
https://www.aepd.es/guias/tech-dispatch-aprendizaje-federado.pdf

> Informe conjunto del regulador español y el europeo dedicado específicamente al aprendizaje
> federado. Verificado descargando el documento original. Reconoce sus beneficios (menor
> concentración de datos, minimización, protección desde el diseño) pero advierte de que **los
> parámetros del modelo intercambiados pueden seguir siendo datos personales**, por ser vulnerables
> a ataques de inferencia, de pertenencia y de inversión del modelo; concluye que el FL no basta por
> sí solo como anonimización y recomienda complementarlo con privacidad diferencial o agregación
> segura. Es la fuente que delimita el alcance de las afirmaciones de privacidad de la memoria y la
> que fundamenta que RQ5 quede como línea futura. Citado por primera vez: Sesión 32.

**IAB Tech Lab. (2024).** Data clean rooms: Guidance and recommended practices (Versión 1.0). IAB
Technology Laboratory. https://iabtechlab.com/datacleanrooms/

> Guía de referencia del sector sobre data clean rooms, verificada en la página oficial del
> estándar. Sistematiza principios, funciones y limitaciones, y acota sus casos de uso a tres
> ámbitos publicitarios: activación de audiencias, enriquecimiento de datos de consumidor y
> medición/optimización de campañas. Sostiene el argumento central de RQ4: los clean rooms están
> diseñados para consultar datos existentes, no para entrenar modelos predictivos entre partes.
> Citado por primera vez: Sesión 32.

---

## Datasets

**Kaggle. (s.f.).** Store Sales - Time Series Forecasting [Conjunto de datos]. Kaggle.
https://www.kaggle.com/competitions/store-sales-time-series-forecasting (Recuperado el 9 de julio de 2026)

> Dataset principal del proyecto: ventas diarias de Corporación Favorita, 54 tiendas, 33 familias de
> producto, 2013–2017. Citado por primera vez: Sesión 2 (adquisición de datos).

**Kaggle. (2017).** Corporación Favorita Grocery Sales Forecasting [Conjunto de datos]. Kaggle.
https://www.kaggle.com/c/favorita-grocery-sales-forecasting (Recuperado el 11 de julio de 2026)

> Competición original (nivel de producto individual, no usada como dataset principal — ver
> `RESEARCH_LOG.md` Sesión 5 sobre la decisión de granularidad familia/semana). Fuente de la métrica
> oficial NWRMSLE, que justifica la transformación logarítmica del objetivo. Citado por primera vez:
> Sesión 5 (justificación de log(1+ventas)).

---

## Plantilla para nuevas entradas

```markdown
**Autor, A. A. (Año).** Título del trabajo. *Fuente/Revista/Editorial*. URL (Recuperado el DD de mes de AAAA, si aplica)

> Para qué se usó esta fuente en el proyecto. Citado por primera vez: Sesión N del RESEARCH_LOG.
```
