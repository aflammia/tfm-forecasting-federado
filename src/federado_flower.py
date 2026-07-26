"""
federado_flower.py — infraestructura de Federated Learning con Flower (T3.1).

Los 3 SILOS son los clientes/participantes de FedAvg (no las 54 tiendas sueltas -- cada silo
centraliza internamente sus propias tiendas, es el mismo operador simulado; Sesión 5). Reutiliza
`MLPConEmbeddings` de `modelo_mlp.py` sin ninguna modificación -- lo único que cambia frente a
A/B/C es CÓMO se entrena: en vez de un único `fit()` con early stopping, aquí cada silo hace
pocas épocas locales por RONDA, y el servidor promedia los pesos (ponderado por nº de filas)
entre rondas. Esta es la diferencia central de FedAvg (McMahan et al., 2017) frente a un
promediado de un solo paso: pocas épocas por ronda, muchas rondas.

Soporta dos estrategias:
  - FedAvg — promediado simple ponderado por nº de filas.
  - FedProx — añade un término proximal a la pérdida local (mu/2 * ||w - w_global||^2) que
    penaliza alejarse demasiado de los pesos globales en cada ronda; diseñado específicamente
    para clientes no-IID (Li et al., 2020) -- relevante aquí porque la Sesión 24 ya encontró
    heterogeneidad no-IID real entre tiendas de un mismo silo.

Checkpointing por ronda: como Flower no guarda el historial de pesos de cada ronda por defecto,
`evaluate_fn` escribe los pesos de esa ronda a disco (`checkpoints_dir`) además de devolver la
pérdida centralizada -- así, terminada la simulación, se puede recuperar la MEJOR ronda (la de
menor pérdida de validación centralizada), igual que `entrenar()` (modelo_mlp.py) se queda con
los mejores pesos vistos, no los últimos.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation

from modelo_mlp import DatasetVentas, MLPConEmbeddings


def get_params(modelo: MLPConEmbeddings) -> list[np.ndarray]:
    """OJO: `.numpy()` sobre un tensor en CPU comparte memoria con el tensor original (zero-copy)
    -- sin `.copy()` explícito, el array "snapshot" devuelto aquí seguiría cambiando en
    silencio si el modelo se sigue entrenando después (p.ej. `optimizer.step()` modifica los
    pesos in-place). Detectado en la Sesión 26 al ver que dos snapshots antes/después de
    entrenar salían idénticos pese a que la pérdida sí bajaba -- la simulación real de Flower no
    lo sufría porque Ray serializa los datos entre procesos (rompiendo el alias de memoria como
    efecto secundario), pero cualquier uso directo en el mismo proceso (tests, checkpointing) sí
    quedaba expuesto."""
    return [v.detach().cpu().numpy().copy() for v in modelo.state_dict().values()]


def set_params(modelo: MLPConEmbeddings, parametros: list[np.ndarray]) -> None:
    claves = modelo.state_dict().keys()
    estado = {k: torch.tensor(v) for k, v in zip(claves, parametros)}
    modelo.load_state_dict(estado, strict=True)


class ClienteSilo(NumPyClient):
    """Un cliente Flower = un silo. `fit()` hace `epocas_locales` épocas (pocas) sobre TODO el
    train del silo en cada ronda -- no un `entrenar()` completo con early stopping, esa es
    precisamente la diferencia entre entrenar de una vez (A/B/C) y FedAvg iterativo (D)."""

    def __init__(self, silo: str, train_df: pd.DataFrame, epocas_locales: int = 1,
                 lr: float = 1e-3, batch_size: int = 256):
        self.silo = silo
        self.epocas_locales = epocas_locales
        self.lr = lr
        self.batch_size = batch_size
        self.modelo = MLPConEmbeddings()
        self.dl_train = DataLoader(DatasetVentas(train_df), batch_size=batch_size, shuffle=True)

    def get_parameters(self, config):
        return get_params(self.modelo)

    def fit(self, parameters, config):
        set_params(self.modelo, parameters)
        mu = float(config.get("proximal_mu", 0.0))
        pesos_globales = [p.detach().clone() for p in self.modelo.parameters()] if mu > 0 else None

        opt = torch.optim.Adam(self.modelo.parameters(), lr=self.lr)
        perdida_fn = nn.HuberLoss()
        for _ in range(self.epocas_locales):
            for x_cont, fam, tienda, y in self.dl_train:
                opt.zero_grad()
                pred = self.modelo(x_cont, fam, tienda)
                perdida = perdida_fn(pred, y)
                if mu > 0:
                    termino_proximal = sum(
                        (p - p_g).pow(2).sum() for p, p_g in zip(self.modelo.parameters(), pesos_globales)
                    )
                    perdida = perdida + (mu / 2) * termino_proximal
                perdida.backward()
                opt.step()

        return get_params(self.modelo), len(self.dl_train.dataset), {"silo": self.silo}

    def evaluate(self, parameters, config):
        set_params(self.modelo, parameters)
        # evaluación local solo informativa (métrica agregada real se hace centralizada, con
        # evaluate_fn, sobre el val global -- más estable, Sesión 24 ya mostró que un val por
        # entidad pequeño es ruidoso).
        return 0.0, len(self.dl_train.dataset), {}


def _hacer_client_fn(datos_por_silo: dict[str, pd.DataFrame], epocas_locales: int):
    silos = sorted(datos_por_silo.keys())

    def client_fn(context: Context):
        idx = context.node_config["partition-id"]
        silo = silos[idx]
        return ClienteSilo(silo, datos_por_silo[silo], epocas_locales=epocas_locales).to_client()

    return client_fn


def _hacer_evaluate_fn(val_global: pd.DataFrame, checkpoints_dir: Path, historial_perdidas: dict):
    """`historial_perdidas` es un dict MUTABLE pasado por el llamador -- `run_simulation()` no
    devuelve nada (a diferencia de la API clásica `start_simulation`/`History`), así que esta es
    la forma de recuperar la pérdida centralizada de cada ronda una vez termina la simulación."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dl_val = DataLoader(DatasetVentas(val_global), batch_size=2048, shuffle=False)
    perdida_fn = nn.HuberLoss()
    modelo_temp = MLPConEmbeddings()

    def evaluate_fn(server_round, parameters, config):
        set_params(modelo_temp, parameters)
        modelo_temp.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x_cont, fam, tienda, y in dl_val:
                pred = modelo_temp(x_cont, fam, tienda)
                total += perdida_fn(pred, y).item() * len(y)
                n += len(y)
        val_loss = total / n
        np.savez(checkpoints_dir / f"ronda_{server_round}.npz", *parameters)
        historial_perdidas[server_round] = val_loss
        return val_loss, {"ronda": server_round}

    return evaluate_fn


def ejecutar_federado(
    datos_por_silo: dict[str, pd.DataFrame],
    val_global: pd.DataFrame,
    checkpoints_dir: Path,
    *,
    estrategia: str = "fedavg",
    proximal_mu: float = 0.0,
    num_rounds: int = 30,
    epocas_locales: int = 2,
    semilla: int = 42,
) -> dict[int, float]:
    """Ejecuta una simulación FedAvg o FedProx sobre los silos dados. Devuelve el historial de
    pérdida centralizada por ronda ({ronda: val_loss}); los pesos de cada ronda quedan en
    `checkpoints_dir` (ver `cargar_mejor_ronda`)."""
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
    torch.manual_seed(semilla)

    n_clientes = len(datos_por_silo)
    client_app = ClientApp(client_fn=_hacer_client_fn(datos_por_silo, epocas_locales))
    historial_perdidas: dict[int, float] = {}
    evaluate_fn = _hacer_evaluate_fn(val_global, checkpoints_dir, historial_perdidas)

    modelo_inicial = MLPConEmbeddings()
    parametros_iniciales = ndarrays_to_parameters(get_params(modelo_inicial))

    kwargs_estrategia = dict(
        fraction_fit=1.0, fraction_evaluate=0.0,  # evaluación centralizada, no distribuida
        min_fit_clients=n_clientes, min_available_clients=n_clientes,
        evaluate_fn=evaluate_fn, initial_parameters=parametros_iniciales,
    )
    if estrategia == "fedavg":
        strategy = FedAvg(**kwargs_estrategia)
    elif estrategia == "fedprox":
        strategy = FedProx(proximal_mu=proximal_mu, **kwargs_estrategia)
    else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")

    def server_fn(context: Context):
        from flwr.compat.server.serverapp_components import ServerAppComponents
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

    server_app = ServerApp(server_fn=server_fn)
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clientes,
                    verbose_logging=False)

    return historial_perdidas


def cargar_mejor_ronda(checkpoints_dir: Path, historial_perdidas: dict[int, float]) -> MLPConEmbeddings:
    mejor_ronda = min(historial_perdidas, key=historial_perdidas.get)
    datos = np.load(checkpoints_dir / f"ronda_{mejor_ronda}.npz")
    parametros = [datos[k] for k in datos.files]
    modelo = MLPConEmbeddings()
    set_params(modelo, parametros)
    return modelo, mejor_ronda
