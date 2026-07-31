"""
T2.3 — Tests de src/modelo_mlp.py (arquitectura MLP+embeddings compartida por A-E).

Usa datos sintéticos pequeños, no dataset_features.parquet -- rápido, y no depende de que el
pipeline de datos ya esté generado en este entorno.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modelo_mlp import (
    FEATURES_CONTINUAS,
    N_FAMILIAS,
    N_TIENDAS,
    ArquitecturaMLP,
    DatasetVentas,
    MLPConEmbeddings,
    entrenar,
    predecir,
)


def _df_sintetico(n: int, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(semilla)
    datos = {c: rng.normal(0, 1, n) for c in FEATURES_CONTINUAS}
    datos["family_id"] = rng.randint(0, N_FAMILIAS, n)
    datos["store_id"] = rng.randint(0, N_TIENDAS, n)
    # señal aprendible: log_ventas depende linealmente de lag_log_1_z + ruido pequeño
    datos["log_ventas"] = 5.0 + 2.0 * datos["lag_log_1_z"] + rng.normal(0, 0.05, n)
    return pd.DataFrame(datos)


# ============================================================ arquitectura

def test_recuento_de_parametros_es_el_documentado():
    """4.985 parámetros -- el número exacto verificado en el diagrama de arquitectura publicado
    (Sesión 22): embeddings 33x8 + 54x8, Densa(33->64), Densa(64->32), Densa(32->1)."""
    modelo = MLPConEmbeddings()
    total = sum(p.numel() for p in modelo.parameters())
    assert total == 4985


def test_forward_devuelve_una_prediccion_por_fila():
    modelo = MLPConEmbeddings()
    n = 10
    x_cont = torch.randn(n, len(FEATURES_CONTINUAS))
    family_id = torch.randint(0, N_FAMILIAS, (n,))
    store_id = torch.randint(0, N_TIENDAS, (n,))
    salida = modelo(x_cont, family_id, store_id)
    assert salida.shape == (n,)


def test_embeddings_tienen_las_dimensiones_documentadas():
    modelo = MLPConEmbeddings()
    assert modelo.emb_familia.weight.shape == (N_FAMILIAS, 8)
    assert modelo.emb_tienda.weight.shape == (N_TIENDAS, 8)


# ============================================================ dataset

def test_dataset_ventas_produce_tensores_del_tamano_correcto():
    df = _df_sintetico(20)
    ds = DatasetVentas(df)
    assert len(ds) == 20
    x_cont, fam, tienda, y = ds[0]
    assert x_cont.shape == (len(FEATURES_CONTINUAS),)
    assert fam.dtype == torch.long
    assert tienda.dtype == torch.long
    assert y.dtype == torch.float32


# ============================================================ entrenamiento

def test_entrenar_reduce_la_perdida_con_señal_aprendible():
    """Con una relación lineal simple y clara (log_ventas ≈ 5 + 2*lag_log_1_z), el modelo debe
    aprenderla: la pérdida de la última época debe ser sustancialmente menor que la primera."""
    train_df = _df_sintetico(300, semilla=1)
    val_df = _df_sintetico(80, semilla=2)
    modelo, hist = entrenar(train_df, val_df, epochs=60, paciencia=60, semilla=42)
    assert hist["train_loss"][-1] < hist["train_loss"][0] * 0.5


def test_entrenar_respeta_early_stopping_y_devuelve_mejores_pesos():
    """Con paciencia muy baja Y un val cuyo objetivo no guarda relación con las features (nada
    que seguir aprendiendo, a diferencia del resto de tests), debe parar mucho antes del tope de
    épocas en vez de agotarlas todas."""
    train_df = _df_sintetico(200, semilla=3)
    val_df = _df_sintetico(60, semilla=4)
    rng = np.random.RandomState(7)
    val_df["log_ventas"] = rng.normal(5.0, 2.0, len(val_df))  # sin relación con las features
    modelo, hist = entrenar(train_df, val_df, epochs=200, paciencia=3, semilla=42)
    assert hist["epocas_entrenadas"] < 200


def test_predecir_nunca_devuelve_ventas_negativas():
    train_df = _df_sintetico(200, semilla=5)
    val_df = _df_sintetico(50, semilla=6)
    modelo, _ = entrenar(train_df, val_df, epochs=10, paciencia=10, semilla=42)
    pred = predecir(modelo, val_df)
    assert (pred >= 0).all()
    assert len(pred) == len(val_df)


# ============================================================ continuar desde pesos dados (E, T3.3)

def test_entrenar_con_modelo_inicial_parte_de_esos_pesos_no_de_aleatorio():
    """Sesión 26 (Condición E): entrenar() debe poder CONTINUAR desde un modelo ya entrenado
    (p.ej. el global convergido de FedAvg/D) en vez de siempre reinicializar al azar. Con
    epochs=0 no hay ninguna actualización de gradiente -> el modelo devuelto debe ser
    EXACTAMENTE el modelo_inicial, no una inicialización aleatoria nueva."""
    train_df = _df_sintetico(50, semilla=20)
    val_df = _df_sintetico(20, semilla=21)
    modelo_previo, _ = entrenar(train_df, val_df, epochs=5, paciencia=5, semilla=1)

    modelo_continuado, hist = entrenar(train_df, val_df, epochs=0, paciencia=1, semilla=99,
                                        modelo_inicial=modelo_previo)

    for p1, p2 in zip(modelo_previo.parameters(), modelo_continuado.parameters()):
        assert torch.allclose(p1, p2)


# ============================================================ hook de tracking (W&B, Fase 3)

class _FalsoWandbRun:
    """Doble de prueba de un wandb.Run -- no depende de tener wandb instalado ni de red."""

    def __init__(self):
        self.llamadas = []

    def log(self, datos: dict):
        self.llamadas.append(datos)


def test_entrenar_llama_a_wandb_run_log_por_cada_epoca():
    train_df = _df_sintetico(150, semilla=8)
    val_df = _df_sintetico(40, semilla=9)
    falso_run = _FalsoWandbRun()
    modelo, hist = entrenar(train_df, val_df, epochs=8, paciencia=8, semilla=42, wandb_run=falso_run)
    assert len(falso_run.llamadas) == hist["epocas_entrenadas"]
    assert all({"train_loss", "val_loss", "epoca"} <= set(reg.keys()) for reg in falso_run.llamadas)


def test_entrenar_sin_wandb_run_no_falla():
    """wandb_run=None (por defecto) no debe intentar llamar a nada."""
    train_df = _df_sintetico(100, semilla=10)
    val_df = _df_sintetico(30, semilla=11)
    modelo, hist = entrenar(train_df, val_df, epochs=5, paciencia=5, semilla=42)
    assert hist["epocas_entrenadas"] > 0


# ============================================================ arquitectura configurable (Sesión 28, tuning)

def test_arquitectura_por_defecto_da_el_recuento_de_parametros_documentado():
    """ArquitecturaMLP() sin argumentos debe reproducir exactamente los 4.985 parámetros
    verificados en el diagrama de arquitectura (Sesión 22) -- ningún cambio de comportamiento
    por defecto tras añadir la configurabilidad."""
    modelo = MLPConEmbeddings(arq=ArquitecturaMLP())
    assert sum(p.numel() for p in modelo.parameters()) == 4985


def test_arquitectura_personalizada_cambia_la_forma_de_la_red():
    arq = ArquitecturaMLP(dim_emb=16, hidden1=128, hidden2=64, dropout=0.3)
    modelo = MLPConEmbeddings(arq=arq)
    assert modelo.emb_familia.weight.shape == (N_FAMILIAS, 16)
    assert modelo.emb_tienda.weight.shape == (N_TIENDAS, 16)
    assert modelo.red[0].out_features == 128
    assert modelo.red[3].out_features == 64
    assert modelo.red[2].p == 0.3


def test_entrenar_propaga_arq_al_crear_el_modelo_desde_cero():
    train_df = _df_sintetico(80, semilla=12)
    val_df = _df_sintetico(24, semilla=13)
    arq = ArquitecturaMLP(dim_emb=4, hidden1=16, hidden2=8, dropout=0.1)
    modelo, _ = entrenar(train_df, val_df, epochs=3, paciencia=3, semilla=42, arq=arq)
    assert modelo.emb_familia.weight.shape == (N_FAMILIAS, 4)
    assert modelo.red[0].out_features == 16


# ============================================================ congelar_base -- personalización FedPer (Sesión 28)

def test_entrenar_con_modelo_inicial_hereda_su_arquitectura_no_la_de_arq():
    """Si modelo_inicial tiene una arquitectura no estándar, entrenar() debe copiarla con
    copy.deepcopy -- no reconstruir con MLPConEmbeddings(arq=...) y cargar el state_dict, que
    fallaría por incompatibilidad de formas si arq no coincide."""
    train_df = _df_sintetico(60, semilla=14)
    val_df = _df_sintetico(20, semilla=15)
    arq_no_estandar = ArquitecturaMLP(dim_emb=4, hidden1=16, hidden2=8)
    modelo_previo, _ = entrenar(train_df, val_df, epochs=2, paciencia=2, semilla=1, arq=arq_no_estandar)

    # arq por defecto (33 dim de entrada por ejemplo) NO debe usarse -- debe heredarse la de modelo_previo
    modelo_continuado, _ = entrenar(train_df, val_df, epochs=0, paciencia=1, semilla=2,
                                     modelo_inicial=modelo_previo)
    assert modelo_continuado.red[0].out_features == 16
    assert modelo_continuado.emb_familia.weight.shape == (N_FAMILIAS, 4)


def test_congelar_base_deja_las_capas_densas_compartidas_sin_cambiar():
    train_df = _df_sintetico(120, semilla=16)
    val_df = _df_sintetico(30, semilla=17)
    modelo_global, _ = entrenar(train_df, val_df, epochs=5, paciencia=5, semilla=1)
    pesos_base_antes = modelo_global.red[0].weight.detach().clone()

    modelo_personalizado, _ = entrenar(
        train_df, val_df, epochs=5, paciencia=5, lr=0.05, semilla=2,
        modelo_inicial=modelo_global, congelar_base=True,
    )
    assert torch.allclose(pesos_base_antes, modelo_personalizado.red[0].weight)


def test_congelar_base_si_permite_que_cambien_embeddings_y_capa_de_salida():
    """red[5] es la capa de salida (red[4] es un ReLU, sin pesos)."""
    train_df = _df_sintetico(120, semilla=18)
    val_df = _df_sintetico(30, semilla=19)
    modelo_global, _ = entrenar(train_df, val_df, epochs=5, paciencia=5, semilla=1)
    salida_antes = modelo_global.red[5].weight.detach().clone()

    modelo_personalizado, _ = entrenar(
        train_df, val_df, epochs=8, paciencia=8, lr=0.05, semilla=2,
        modelo_inicial=modelo_global, congelar_base=True,
    )
    assert not torch.allclose(salida_antes, modelo_personalizado.red[5].weight)


def test_congelar_base_sin_modelo_inicial_lanza_error():
    train_df = _df_sintetico(50, semilla=20)
    val_df = _df_sintetico(20, semilla=21)
    with pytest.raises(ValueError):
        entrenar(train_df, val_df, epochs=1, congelar_base=True)
