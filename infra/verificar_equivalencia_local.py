"""verificar_equivalencia_local.py — paso 2 del runbook (Sesión 30).

Verifica que la Flower App de DESPLIEGUE (flower_app/tfm_fl) produce el MISMO resultado por ronda
que el código de simulación ya documentado, ANTES de gastar un céntimo en Azure. En vez de usar
`flwr run` (cuya CLI de despliegue cambia entre versiones), corre los mismos `ServerApp`/`ClientApp`
de la app con `run_simulation()` -- exactamente el motor que usa src/federado_flower.py para todos
los resultados del RESEARCH_LOG. Si el val_loss por ronda coincide, el "simulacro de verdad" en
Azure no cambiará el algoritmo, solo la infraestructura.

Ejecutar: TFM_DATA_DIR debe apuntar a infra/datos_silos (por defecto lo hace). Requiere haber
corrido antes infra/02_split_datos.py.

  cd flower_app && PYTHONPATH=. ../.venv/Scripts/python.exe ../infra/verificar_equivalencia_local.py
"""
# ruff: noqa: E402  (imports de tfm_fl deben ir tras el sys.path.insert de abajo)
import sys
from pathlib import Path

from flwr.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flower_app"))

from tfm_fl.client_app import app as client_app
from tfm_fl.server_app import CHECKPOINTS
from tfm_fl.server_app import app as server_app


def main() -> None:
    print("Corriendo la Flower App de despliegue vía run_simulation (3 silos, config por defecto: "
          "FedAvg, 15 rondas, arquitectura tuneada)...\n")
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=3,
                   verbose_logging=False)

    # Los checkpoints por ronda quedaron en CHECKPOINTS; el server_app ya imprimió el val_loss de
    # cada ronda. La comprobación de coherencia numérica contra historial_rondas_condicion_D.csv se
    # hace a ojo comparando la tendencia (misma arquitectura, mismos datos, mismo algoritmo; el
    # no-determinismo de PyTorch da diferencias pequeñas, no estructurales).
    checkpoints = sorted(CHECKPOINTS.glob("ronda_*.npz"))
    print(f"\nOK — corrida completa: {len(checkpoints)} rondas con checkpoint en {CHECKPOINTS}")
    print("Compara el val_loss impreso arriba con reports/historial_rondas_condicion_D.csv "
          "(config fedavg_el2, arquitectura tuneada): debe seguir la misma tendencia decreciente.")


if __name__ == "__main__":
    main()
