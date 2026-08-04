"""06_registrar_modelo.py — registra el modelo global del simulacro en el registro de Azure ML.

Corre EN EL COORDINADOR tras la(s) corrida(s): tiene los checkpoints por ronda, el val_global de
referencia y la managed identity (DefaultAzureCredential autentica sin secretos). Reevalúa cada
checkpoint sobre val_global, elige la mejor ronda (menor val_loss), guarda su state_dict y lo
registra como Model en el workspace con métricas como tags. Cierra la pieza de "registro de
modelos en Azure ML" de la Fase 4, ahora con el modelo REAL producido por el simulacro distribuido.

Uso (en el coordinador):  source ~/venv/bin/activate && source ~/tfm_env.sh && \
    python ~/tfm/flower_app/../infra/06_registrar_modelo.py   # o la ruta donde esté el repo
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TFM_REPO_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "flower_app"))

from tfm_fl.task import cargar_arquitectura, cargar_val_global  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from federado_flower import set_params  # noqa: E402
from metrics import wmape  # noqa: E402
from modelo_mlp import DatasetVentas, MLPConEmbeddings, predecir  # noqa: E402

CHECKPOINTS = Path(os.environ.get("TFM_CHECKPOINTS_DIR", str(ROOT / "infra" / "checkpoints_despliegue")))
SALIDA = ROOT / "infra" / "modelo_registrado"


def _mejor_ronda(arq, val_global):
    """Reevalúa cada checkpoint sobre val_global y devuelve (ronda, val_loss, wmape, parametros) de
    la mejor. La evaluación es idéntica a la del server_app -- así el modelo registrado es exactamente
    el mejor punto visto durante el entrenamiento federado."""
    dl_val = DataLoader(DatasetVentas(val_global), batch_size=2048, shuffle=False)
    perdida_fn = nn.HuberLoss()
    modelo = MLPConEmbeddings(arq=arq)
    mejor = None
    for ckpt in sorted(CHECKPOINTS.glob("ronda_*.npz")):
        ronda = int(ckpt.stem.split("_")[1])
        datos = np.load(ckpt)
        parametros = [datos[k] for k in datos.files]
        set_params(modelo, parametros)
        modelo.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x_cont, fam, tienda, y in dl_val:
                total += perdida_fn(modelo(x_cont, fam, tienda), y).item() * len(y)
                n += len(y)
        val_loss = total / n
        if mejor is None or val_loss < mejor[1]:
            w = wmape(val_global["ventas"].values, predecir(modelo, val_global))
            mejor = (ronda, val_loss, w, parametros)
    return mejor


def main() -> None:
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.identity import DefaultAzureCredential

    arq = cargar_arquitectura()
    val_global = cargar_val_global()
    ronda, val_loss, w, parametros = _mejor_ronda(arq, val_global)
    print(f"Mejor ronda: {ronda}  val_loss={val_loss:.5f}  wmape_val={w:.4f}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    modelo = MLPConEmbeddings(arq=arq)
    set_params(modelo, parametros)
    ruta_pesos = SALIDA / "modelo_global_federado.pt"
    torch.save(modelo.state_dict(), ruta_pesos)
    print(f"state_dict guardado en {ruta_pesos}")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZ_SUBSCRIPTION"],
        resource_group_name=os.environ["AZ_RG"],
        workspace_name=os.environ["AZ_WORKSPACE"],
    )
    modelo_azure = Model(
        path=str(ruta_pesos),
        name="tfm-federado-global",
        type=AssetTypes.CUSTOM_MODEL,
        description=("Modelo global MLP+embeddings del simulacro federado real (3 silos, "
                     "Deployment Engine de Flower en Azure). Ver docs/RESEARCH_LOG.md Sesión 30."),
        tags={
            "estrategia": os.environ.get("TFM_ESTRATEGIA", "fedavg"),
            "mejor_ronda": str(ronda),
            "val_loss": f"{val_loss:.5f}",
            "wmape_val": f"{w:.4f}",
            "arquitectura": f"{arq.hidden1}x{arq.hidden2}_emb{arq.dim_emb}_do{arq.dropout}",
            "sesion_research_log": "30",
        },
    )
    registrado = ml_client.models.create_or_update(modelo_azure)
    print(f"OK — modelo registrado en Azure ML: {registrado.name} v{registrado.version}")


if __name__ == "__main__":
    main()
