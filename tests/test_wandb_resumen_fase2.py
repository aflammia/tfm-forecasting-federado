"""
Tests de src/15_wandb_resumen_fase2.py — solo la parte de carga/normalización de CSVs (no
dispara ninguna corrida de W&B; eso se prueba manualmente con `WANDB_MODE=offline`, Sesión 25).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from importlib import import_module

modulo = import_module("15_wandb_resumen_fase2")


def test_cargar_resumen_fase2_une_los_5_ficheros_con_columna_metodo_comun():
    resumen = modulo.cargar_resumen_fase2()
    assert "metodo" in resumen.columns
    assert "baseline" not in resumen.columns
    assert "condicion" not in resumen.columns
    # 8 métodos distintos (3 ingenuos + 2 convencionales + 3 condiciones A/B/C)
    assert resumen["metodo"].nunique() == 8


def test_cargar_resumen_fase2_incluye_val_y_test_para_cada_metodo():
    resumen = modulo.cargar_resumen_fase2()
    for metodo, grupo in resumen.groupby("metodo"):
        assert set(grupo["split"]) == {"val", "test"}
