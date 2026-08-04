"""task.py — utilidades compartidas por client_app y server_app.

Reutiliza el modelo y las funciones de (de)serialización de pesos ya escritas y probadas en
src/, en vez de duplicarlas -- así el simulacro distribuido usa EXACTAMENTE el mismo MLP y la
misma lógica que la simulación local documentada en el RESEARCH_LOG. Solo cambia de dónde salen
los datos: aquí cada silo lee su propio parquet local (aislamiento real), no una partición de un
DataFrame en memoria.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

# --- reutilización de src/ (mismo patrón que los scripts y los tests del proyecto) ---
# En local (simulación de equivalencia), la raíz del repo es dos niveles por encima de este
# fichero. En las VMs de Azure, el código corre empaquetado en un FAB de Flower fuera del repo, así
# que la raíz se pasa explícitamente por TFM_REPO_ROOT (la ruta donde cloud-init clonó el repo).
ROOT = Path(os.environ.get("TFM_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "src"))

from modelo_mlp import ArquitecturaMLP, MLPConEmbeddings  # noqa: E402
from federado_flower import get_params, set_params  # noqa: E402

__all__ = ["ArquitecturaMLP", "MLPConEmbeddings", "get_params", "set_params",
           "cargar_arquitectura", "cargar_datos_silo", "cargar_val_global", "resolver_silo",
           "SILOS"]

# Orden canónico de los silos -- usado para mapear partition-id -> silo en la simulación local
# (paso 2 del runbook). En el despliegue real cada SuperNode trae su silo por node-config y este
# orden no se usa.
SILOS = ["Grande", "Mediano", "Pequeno"]


def _dir_datos() -> Path:
    """Carpeta donde viven los parquets de datos. En local (simulación de equivalencia) apunta a
    infra/datos_silos/; en cada VM se sobreescribe con la env var TFM_DATA_DIR (donde la VM tenga
    montado SOLO su propio parquet)."""
    return Path(os.environ.get("TFM_DATA_DIR", str(ROOT / "infra" / "datos_silos")))


def cargar_arquitectura() -> ArquitecturaMLP:
    """Arquitectura ganadora del tuning (Sesión 28, configs/mlp_arquitectura.json). Fallback a la
    de fábrica si el fichero no está -- así la app sigue siendo ejecutable de forma independiente."""
    ruta = ROOT / "configs" / "mlp_arquitectura.json"
    if not ruta.exists():
        return ArquitecturaMLP()
    cfg = json.loads(ruta.read_text(encoding="utf-8"))
    return ArquitecturaMLP(dim_emb=cfg["dim_emb"], hidden1=cfg["hidden1"],
                           hidden2=cfg["hidden2"], dropout=cfg["dropout"])


def resolver_silo(node_config: dict) -> str:
    """Determina qué silo es este cliente. En DESPLIEGUE real, cada SuperNode se arranca con
    `--node-config "silo='Grande'"` y ese valor manda. En SIMULACIÓN local (paso 2), Flower asigna
    `partition-id` 0/1/2 a los SuperNodes y se mapea al orden canónico de SILOS -- así la MISMA app
    corre en ambos entornos sin cambios."""
    if "silo" in node_config:
        return str(node_config["silo"])
    idx = int(node_config["partition-id"])
    return SILOS[idx]


def cargar_datos_silo(silo: str) -> pd.DataFrame:
    """Lee el parquet de train de ESTE silo (y solo el suyo). En una VM de silo, TFM_DATA_DIR
    contiene únicamente este fichero -- el aislamiento es físico, no solo lógico."""
    ruta = _dir_datos() / f"{silo.lower()}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra el parquet del silo {silo} en {ruta}. En local, ejecuta primero "
            f"infra/02_split_datos.py; en una VM, revisa TFM_DATA_DIR e infra/03_deploy_datos.sh."
        )
    return pd.read_parquet(ruta)


def cargar_val_global() -> pd.DataFrame:
    """Set de validación de referencia para la evaluación centralizada del coordinador. Solo el
    coordinador tiene este fichero (los silos no lo necesitan ni lo reciben)."""
    ruta = _dir_datos() / "val_global.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra val_global.parquet en {ruta}. En local, ejecuta infra/02_split_datos.py."
        )
    return pd.read_parquet(ruta)
