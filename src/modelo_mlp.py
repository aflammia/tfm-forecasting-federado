"""
modelo_mlp.py — arquitectura MLP + embeddings compartida por las condiciones A-E (T2.3-T3.3).

Misma red para las 5 condiciones experimentales (Sesión 5 del RESEARCH_LOG); lo que cambia entre
condiciones es qué datos la entrenan, no su forma:

  17 features continuas (ya estandarizadas, T1.4b) + embedding de family_id (33->8) +
  embedding de store_id (54->8) -> concatenación (33 dim) -> Densa(33->64) -> ReLU ->
  Dropout(0,2) -> Densa(64->32) -> ReLU -> Densa(32->1).

Pérdida Huber sobre log(1+ventas) (robusta al pico del terremoto de abril 2016, Sesión 5).
Optimizador Adam. 4.985 parámetros (recuento exacto verificado, no la estimación aproximada
original de la Sesión 5).
"""
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

FEATURES_CONTINUAS = [
    "lag_log_1_z", "lag_log_2_z", "lag_log_4_z", "lag_log_8_z",
    "media_movil_log_4_z", "media_movil_log_8_z", "std_log_4_z",
    "log_onpromotion_z", "oil_price_z",
    "semana_sin", "semana_cos", "mes_sin", "mes_cos",
    "es_festivo_nacional", "es_festivo_regional", "es_festivo_local", "semana_con_dia_pago",
]
N_FAMILIAS = 33
N_TIENDAS = 54
DIM_EMBEDDING = 8


class DatasetVentas(Dataset):
    """Envuelve un DataFrame de dataset_features.parquet en tensores listos para el MLP."""

    def __init__(self, df: pd.DataFrame):
        self.x_cont = torch.tensor(
            df[FEATURES_CONTINUAS].astype("float32").values, dtype=torch.float32
        )
        self.family_id = torch.tensor(df["family_id"].values, dtype=torch.long)
        self.store_id = torch.tensor(df["store_id"].values, dtype=torch.long)
        self.y = torch.tensor(df["log_ventas"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cont[idx], self.family_id[idx], self.store_id[idx], self.y[idx]


class MLPConEmbeddings(nn.Module):
    """33 -> 64 -> 32 -> 1. Ver docstring del módulo y RESEARCH_LOG Sesión 5 / 22 (diagrama de
    arquitectura publicado como artefacto) para la justificación de cada elección de diseño."""

    def __init__(self, n_familias: int = N_FAMILIAS, n_tiendas: int = N_TIENDAS,
                 dim_emb: int = DIM_EMBEDDING, n_continuas: int = len(FEATURES_CONTINUAS)):
        super().__init__()
        self.emb_familia = nn.Embedding(n_familias, dim_emb)
        self.emb_tienda = nn.Embedding(n_tiendas, dim_emb)
        entrada = n_continuas + 2 * dim_emb
        self.red = nn.Sequential(
            nn.Linear(entrada, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cont: torch.Tensor, family_id: torch.Tensor, store_id: torch.Tensor) -> torch.Tensor:
        emb_f = self.emb_familia(family_id)
        emb_t = self.emb_tienda(store_id)
        x = torch.cat([x_cont, emb_f, emb_t], dim=1)
        return self.red(x).squeeze(-1)


def entrenar(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    epochs: int = 150,
    paciencia: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    semilla: int = 42,
    verbose: bool = False,
    wandb_run: Optional[object] = None,
) -> tuple[MLPConEmbeddings, dict]:
    """Entrena un MLPConEmbeddings (Huber loss, Adam) con early stopping sobre val_loss.
    Devuelve el modelo con los MEJORES pesos vistos (no los últimos, que pueden estar ya
    sobreajustando) y un historial de pérdidas por época.

    `wandb_run` es opcional y genérico (cualquier objeto con `.log(dict)`, típicamente el valor
    de retorno de `wandb.init()`) -- este módulo no importa `wandb` directamente para no
    forzarlo como dependencia dura de la arquitectura; quien llama a `entrenar()` decide si
    trackea la corrida o no."""
    torch.manual_seed(semilla)
    modelo = MLPConEmbeddings()
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    perdida_fn = nn.HuberLoss()

    dl_train = DataLoader(DatasetVentas(train_df), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(DatasetVentas(val_df), batch_size=1024, shuffle=False)

    mejor_val, mejor_estado, sin_mejora = np.inf, None, 0
    historial = {"train_loss": [], "val_loss": []}
    epoca = 0

    for epoca in range(epochs):
        modelo.train()
        perdida_epoca = 0.0
        for x_cont, fam, tienda, y in dl_train:
            opt.zero_grad()
            pred = modelo(x_cont, fam, tienda)
            l = perdida_fn(pred, y)
            l.backward()
            opt.step()
            perdida_epoca += l.item() * len(y)
        perdida_epoca /= len(dl_train.dataset)

        modelo.eval()
        val_perdida = 0.0
        with torch.no_grad():
            for x_cont, fam, tienda, y in dl_val:
                pred = modelo(x_cont, fam, tienda)
                val_perdida += perdida_fn(pred, y).item() * len(y)
        val_perdida /= len(dl_val.dataset)

        historial["train_loss"].append(perdida_epoca)
        historial["val_loss"].append(val_perdida)
        if wandb_run is not None:
            wandb_run.log({"train_loss": perdida_epoca, "val_loss": val_perdida, "epoca": epoca})

        if val_perdida < mejor_val - 1e-5:
            mejor_val = val_perdida
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

        if verbose and epoca % 10 == 0:
            print(f"    época {epoca}: train={perdida_epoca:.4f} val={val_perdida:.4f}")

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    historial["mejor_val_loss"] = mejor_val
    historial["epocas_entrenadas"] = epoca + 1
    return modelo, historial


def predecir(modelo: MLPConEmbeddings, df: pd.DataFrame) -> np.ndarray:
    """Predicciones en ESCALA NATURAL de ventas (expm1 de la salida en log_ventas)."""
    modelo.eval()
    ds = DatasetVentas(df)
    with torch.no_grad():
        pred_log = modelo(ds.x_cont, ds.family_id, ds.store_id).numpy()
    return np.clip(np.expm1(pred_log), 0, None)
