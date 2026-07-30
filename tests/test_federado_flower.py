"""
T3.1 — Tests de src/federado_flower.py.

No lanza ninguna simulación real de Flower (lenta, minutos) -- prueba directamente las piezas
reutilizables: el round-trip de parámetros, el cliente (`ClienteSilo.fit()`/`.evaluate()`
llamados directamente, sin pasar por `run_simulation`), y la selección de la mejor ronda a
partir de checkpoints en disco.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modelo_mlp import ArquitecturaMLP, FEATURES_CONTINUAS, N_FAMILIAS, N_TIENDAS, MLPConEmbeddings
from federado_flower import ClienteSilo, get_params, set_params, cargar_mejor_ronda


def _df_sintetico(n: int, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(semilla)
    datos = {c: rng.normal(0, 1, n) for c in FEATURES_CONTINUAS}
    datos["family_id"] = rng.randint(0, N_FAMILIAS, n)
    datos["store_id"] = rng.randint(0, N_TIENDAS, n)
    datos["log_ventas"] = 5.0 + 2.0 * datos["lag_log_1_z"] + rng.normal(0, 0.05, n)
    return pd.DataFrame(datos)


# ============================================================ get/set params

def test_get_set_params_es_un_roundtrip_exacto():
    modelo_a = MLPConEmbeddings()
    parametros = get_params(modelo_a)
    modelo_b = MLPConEmbeddings()
    set_params(modelo_b, parametros)
    for p_a, p_b in zip(modelo_a.parameters(), modelo_b.parameters()):
        assert torch.allclose(p_a, p_b)


# ============================================================ ClienteSilo

def test_cliente_silo_fit_devuelve_el_numero_correcto_de_filas():
    train_df = _df_sintetico(80, semilla=1)
    cliente = ClienteSilo("TestSilo", train_df, epocas_locales=1)
    parametros_iniciales = get_params(cliente.modelo)
    _, n_filas, metricas = cliente.fit(parametros_iniciales, config={})
    assert n_filas == 80
    assert metricas["silo"] == "TestSilo"


def test_cliente_silo_fit_actualiza_los_pesos():
    train_df = _df_sintetico(100, semilla=2)
    cliente = ClienteSilo("TestSilo", train_df, epocas_locales=2, lr=0.05)
    parametros_iniciales = get_params(cliente.modelo)
    parametros_nuevos, _, _ = cliente.fit(parametros_iniciales, config={})
    diferencias = [np.abs(a - b).sum() for a, b in zip(parametros_iniciales, parametros_nuevos)]
    assert sum(diferencias) > 0


def test_fedprox_con_mu_alto_mantiene_los_pesos_mas_cerca_del_global():
    """El término proximal penaliza alejarse de los pesos globales -- con mu muy alto, tras las
    mismas épocas, los pesos deben moverse MENOS que con mu=0 (FedAvg puro)."""
    train_df = _df_sintetico(150, semilla=3)

    cliente_fedavg = ClienteSilo("S", train_df, epocas_locales=3, lr=0.05)
    parametros_iniciales = get_params(cliente_fedavg.modelo)
    parametros_fedavg, _, _ = cliente_fedavg.fit(parametros_iniciales, config={"proximal_mu": 0.0})
    movimiento_fedavg = sum(np.abs(a - b).sum() for a, b in zip(parametros_iniciales, parametros_fedavg))

    cliente_fedprox = ClienteSilo("S", train_df, epocas_locales=3, lr=0.05)
    set_params(cliente_fedprox.modelo, parametros_iniciales)
    parametros_fedprox, _, _ = cliente_fedprox.fit(parametros_iniciales, config={"proximal_mu": 50.0})
    movimiento_fedprox = sum(np.abs(a - b).sum() for a, b in zip(parametros_iniciales, parametros_fedprox))

    assert movimiento_fedprox < movimiento_fedavg


# ============================================================ selección de la mejor ronda

def test_cargar_mejor_ronda_selecciona_la_ronda_de_menor_perdida(tmp_path):
    modelo = MLPConEmbeddings()
    parametros = get_params(modelo)
    for ronda in range(3):
        np.savez(tmp_path / f"ronda_{ronda}.npz", *parametros)
    historial = {0: 0.5, 1: 0.2, 2: 0.3}  # la ronda 1 es la mejor

    _, mejor_ronda = cargar_mejor_ronda(tmp_path, historial)
    assert mejor_ronda == 1


# ============================================================ arquitectura configurable (Sesión 28)

def test_cliente_silo_propaga_arq_al_modelo_interno():
    arq = ArquitecturaMLP(dim_emb=16, hidden1=96, hidden2=48)
    cliente = ClienteSilo("S", _df_sintetico(30), arq=arq)
    assert cliente.modelo.emb_familia.weight.shape == (N_FAMILIAS, 16)
    assert cliente.modelo.red[0].out_features == 96


def test_cargar_mejor_ronda_con_arquitectura_no_estandar(tmp_path):
    arq = ArquitecturaMLP(dim_emb=16, hidden1=96, hidden2=48)
    modelo = MLPConEmbeddings(arq=arq)
    parametros = get_params(modelo)
    np.savez(tmp_path / "ronda_0.npz", *parametros)

    modelo_cargado, _ = cargar_mejor_ronda(tmp_path, {0: 0.1}, arq=arq)
    assert modelo_cargado.red[0].out_features == 96
