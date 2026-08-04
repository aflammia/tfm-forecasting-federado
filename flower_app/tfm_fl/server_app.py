"""server_app.py — el ServerApp que corre en el coordinador (SuperLink).

Reutiliza la estrategia FedAvg/FedProx y las funciones de pesos de src/. Lo propio del despliegue:
  1. La evaluación centralizada por ronda se loguea a MLflow -> Azure ML Studio = el DASHBOARD
     donde se ve la métrica mejorar ronda a ronda (si MLFLOW_TRACKING_URI está configurado; si no
     -- p.ej. en la simulación local del paso 2 -- solo se imprime, sin fallar).
  2. Los pesos de cada ronda se guardan a disco para que infra/06_registrar_modelo.py registre
     después la mejor ronda en el registro de modelos de Azure ML.

El coordinador es NEUTRAL: solo ve pesos agregados y un set de validación de referencia
(val_global). Nunca ve las filas de train de ningún silo.
"""
import os
from pathlib import Path

import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from torch import nn
from torch.utils.data import DataLoader

# Importar task PRIMERO: añade ROOT/src al sys.path (honrando TFM_REPO_ROOT) para los imports de src.
from tfm_fl.task import cargar_arquitectura, cargar_val_global, ROOT
from modelo_mlp import DatasetVentas, MLPConEmbeddings, predecir  # noqa: E402
from federado_flower import get_params, set_params  # noqa: E402
from metrics import wmape  # noqa: E402

CHECKPOINTS = Path(os.environ.get(
    "TFM_CHECKPOINTS_DIR", str(ROOT / "infra" / "checkpoints_despliegue"),
))


class _Dashboard:
    """Envoltorio de MLflow que hace no-op si MLflow no está configurado. Así el MISMO server_app
    corre en la simulación local (sin dashboard, solo prints -- paso 2 de equivalencia) y en el
    coordinador de Azure (MLFLOW_TRACKING_URI apuntando al workspace = dashboard en vivo)."""

    def __init__(self, params: dict):
        self._mlflow = None
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            print("[dashboard] MLFLOW_TRACKING_URI no configurado -- solo se imprimen métricas "
                  "(modo simulación local, sin dashboard).")
            return
        try:
            import mlflow
        except ImportError:
            print("[dashboard] mlflow no instalado -- solo prints.")
            return
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("tfm-federado-simulacro")
        self._mlflow = mlflow
        run_name = f"{params.get('estrategia', 'fedavg')}-{params.get('num_rounds', '?')}rondas"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        print(f"[dashboard] logueando a Azure ML como run '{run_name}'.")

    def log(self, metricas: dict, ronda: int):
        if self._mlflow is not None:
            self._mlflow.log_metrics(metricas, step=ronda)


def _hacer_evaluate_fn(dashboard: _Dashboard, arq):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    val_global = cargar_val_global()
    dl_val = DataLoader(DatasetVentas(val_global), batch_size=2048, shuffle=False)
    perdida_fn = nn.HuberLoss()
    modelo_temp = MLPConEmbeddings(arq=arq)

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
        w = wmape(val_global["ventas"].values, predecir(modelo_temp, val_global))

        np.savez(CHECKPOINTS / f"ronda_{server_round}.npz", *parameters)
        dashboard.log({"val_loss": val_loss, "wmape_val": w}, ronda=server_round)
        print(f"[coordinador] ronda {server_round}: val_loss={val_loss:.5f} wmape_val={w:.4f}")
        return val_loss, {"wmape_val": w}

    return evaluate_fn


def server_fn(context: Context) -> ServerAppComponents:
    if CHECKPOINTS.exists():
        import shutil
        shutil.rmtree(CHECKPOINTS)

    num_rounds = int(context.run_config.get("num-server-rounds", 15))
    estrategia = str(context.run_config.get("estrategia", "fedavg")).lower()
    proximal_mu = float(context.run_config.get("proximal-mu", 0.01))
    lr = float(context.run_config.get("lr", 0.002))
    local_epochs = int(context.run_config.get("local-epochs", 2))
    arq = cargar_arquitectura()

    dashboard = _Dashboard({
        "estrategia": estrategia, "num_rounds": num_rounds, "lr": lr,
        "local_epochs": local_epochs, "proximal_mu": proximal_mu if estrategia == "fedprox" else 0,
        "arq_hidden1": arq.hidden1, "arq_hidden2": arq.hidden2, "arq_dropout": arq.dropout,
    })

    modelo_inicial = MLPConEmbeddings(arq=arq)
    parametros_iniciales = ndarrays_to_parameters(get_params(modelo_inicial))

    kwargs = dict(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=3, min_available_clients=3,
        evaluate_fn=_hacer_evaluate_fn(dashboard, arq),
        initial_parameters=parametros_iniciales,
    )
    if estrategia == "fedavg":
        strategy = FedAvg(**kwargs)
    elif estrategia == "fedprox":
        strategy = FedProx(proximal_mu=proximal_mu, **kwargs)
    else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")

    print(f"[coordinador] estrategia={estrategia} rondas={num_rounds} lr={lr} "
          f"epocas_locales={local_epochs} arq={arq}")
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


app = ServerApp(server_fn=server_fn)
