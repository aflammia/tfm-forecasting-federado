"""
Etapa 4 (Sesión 28) — Tests de src/correccion_sesgo.py (estimador de smearing de Duan, 1983).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from correccion_sesgo import aplicar_correccion_duan, factor_correccion_duan


def test_factor_es_uno_si_las_predicciones_son_perfectas():
    """Sin residuo (predicción = real siempre), exp(0)=1 en todos los casos -> factor=1 ->
    la corrección no debe cambiar nada."""
    pred_log = np.array([1.0, 2.0, 3.0, 0.5])
    factor = factor_correccion_duan(log_ventas_real=pred_log, pred_log=pred_log)
    assert np.isclose(factor, 1.0)


def test_factor_coincide_con_la_formula_lognormal_conocida():
    """Si el residuo ~ Normal(0, sigma), E[exp(residuo)] = exp(sigma^2/2) (momento de una
    lognormal) -- se verifica con una muestra grande y una tolerancia razonable."""
    rng = np.random.RandomState(0)
    sigma = 0.4
    n = 200_000
    pred_log = rng.normal(5.0, 1.0, n)
    residuo = rng.normal(0, sigma, n)
    log_real = pred_log + residuo

    factor = factor_correccion_duan(log_real, pred_log)
    esperado = np.exp(sigma ** 2 / 2)
    assert abs(factor - esperado) / esperado < 0.02  # <2% de error de muestreo


def test_factor_mayor_que_uno_si_hay_dispersion_en_el_residuo():
    """Con cualquier dispersión real en el residuo (Jensen), el factor debe ser > 1 -- confirma
    que la corrección "sube" la predicción, no la baja ni la deja igual."""
    rng = np.random.RandomState(1)
    pred_log = np.full(5000, 3.0)
    log_real = pred_log + rng.normal(0, 0.3, 5000)
    factor = factor_correccion_duan(log_real, pred_log)
    assert factor > 1.0


def test_aplicar_correccion_con_factor_uno_no_cambia_la_prediccion():
    pred_log = np.array([1.0, 2.0, 3.5, 0.0])
    corregida = aplicar_correccion_duan(pred_log, factor=1.0)
    esperado = np.expm1(pred_log)
    assert np.allclose(corregida, esperado)


def test_aplicar_correccion_con_factor_mayor_que_uno_aumenta_la_prediccion():
    pred_log = np.array([1.0, 2.0, 3.5])
    sin_corregir = np.expm1(pred_log)
    corregida = aplicar_correccion_duan(pred_log, factor=1.2)
    assert (corregida > sin_corregir).all()


def test_aplicar_correccion_nunca_da_negativo():
    pred_log = np.array([-5.0, -3.0, 0.0])
    corregida = aplicar_correccion_duan(pred_log, factor=0.5)
    assert (corregida >= 0).all()


def test_correccion_reduce_el_sesgo_sistematico_de_infraestimacion():
    """Caso end-to-end: un modelo que predice bien la MEDIANA en escala log pero cuya
    retransformación ingenua (expm1 directo) subestima la MEDIA en escala natural (Jensen) --
    la corrección de Duan debe acercar la predicción agregada a la venta real agregada."""
    rng = np.random.RandomState(2)
    n = 50_000
    pred_log_train = rng.normal(4.0, 1.0, n)
    log_real_train = pred_log_train + rng.normal(0, 0.5, n)  # residuo con dispersión real

    ventas_reales = np.expm1(log_real_train)
    pred_sin_corregir = np.expm1(pred_log_train)
    factor = factor_correccion_duan(log_real_train, pred_log_train)
    pred_corregida = aplicar_correccion_duan(pred_log_train, factor)

    sesgo_sin_corregir = abs(pred_sin_corregir.mean() - ventas_reales.mean())
    sesgo_corregido = abs(pred_corregida.mean() - ventas_reales.mean())
    assert sesgo_corregido < sesgo_sin_corregir
