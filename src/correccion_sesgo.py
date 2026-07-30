"""
correccion_sesgo.py — Etapa 4 del plan de tuning (Sesión 28): corrección del sesgo de Jensen al
deshacer la transformación log1p/expm1.

Un modelo que minimiza error en escala log(1+ventas) y luego deshace con expm1 para reportar en
escala natural subestima sistemáticamente la media condicional (desigualdad de Jensen: para una
variable X con varianza positiva, E[exp(X)] > exp(E[X]) -- así que usar exp(predicción puntual)
como estimador de E[ventas] es, en promedio, demasiado bajo).

El "smearing estimate" de Duan (1983) corrige esto SIN asumir una distribución paramétrica del
residuo: multiplica la predicción retransformada por la media empírica de exp(residuo_log),
calculada sobre TRAIN (nunca val/test, para no filtrar información).
"""
import numpy as np


def factor_correccion_duan(log_ventas_real: np.ndarray, pred_log: np.ndarray) -> float:
    """Factor de Duan -- media de exp(residuo), residuo = real - predicho, ambos en escala log.
    Se calcula UNA VEZ sobre train y se reutiliza igual para val y test (como los estadísticos de
    normalización de T1.4b -- nunca se recalcula con datos de evaluación)."""
    residuo = np.asarray(log_ventas_real) - np.asarray(pred_log)
    return float(np.mean(np.exp(residuo)))


def aplicar_correccion_duan(pred_log: np.ndarray, factor: float) -> np.ndarray:
    """Aplica el factor de Duan a una predicción en escala log, devolviendo ventas en escala
    natural ya corregidas.

    E[ventas] = E[expm1(log_ventas)] = E[exp(log_ventas)] - 1
              ≈ exp(pred_log) * factor - 1
              = (expm1(pred_log) + 1) * factor - 1
    """
    pred_log = np.asarray(pred_log)
    return np.clip((np.expm1(pred_log) + 1) * factor - 1, 0, None)
