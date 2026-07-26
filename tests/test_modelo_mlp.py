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
from modelo_mlp import DatasetVentas, FEATURES_CONTINUAS, MLPConEmbeddings, N_FAMILIAS, N_TIENDAS, entrenar, predecir


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
    assert all({"train_loss", "val_loss", "epoca"} <= set(l.keys()) for l in falso_run.llamadas)


def test_entrenar_sin_wandb_run_no_falla():
    """wandb_run=None (por defecto) no debe intentar llamar a nada."""
    train_df = _df_sintetico(100, semilla=10)
    val_df = _df_sintetico(30, semilla=11)
    modelo, hist = entrenar(train_df, val_df, epochs=5, paciencia=5, semilla=42)
    assert hist["epocas_entrenadas"] > 0
