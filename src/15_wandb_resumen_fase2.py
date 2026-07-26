"""
15_wandb_resumen_fase2.py — integración de Weights & Biases (previa a la Fase 3).

No reentrena nada: registra en W&B el resumen ya calculado de los 8 métodos cerrados en la
Fase 2 (T2.2, T2.2b, T2.3, T2.4, T2.5) como una única corrida de referencia -- una tabla
comparable y un gráfico de barras (WMAPE mediana en test) que sirve de línea base visual antes
de que la Fase 3 (federado) empiece a generar muchas más corridas que comparar.

Modo por defecto: OFFLINE (no requiere cuenta ni login) -- los datos quedan guardados en
`wandb/` (gitignored) y se pueden sincronizar más tarde con `wandb sync <carpeta>` en cuanto haya
una cuenta de W&B configurada (`wandb login`). Fuerza modo online con `WANDB_MODE=online` si ya
se ha hecho login.
"""
import os
from pathlib import Path

import pandas as pd
import wandb

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# (fichero, nombre de la columna del método, fase/tarea de origen)
FUENTES = [
    ("resultados_baselines.csv", "baseline", "T2.2"),
    ("resultados_baselines_convencionales.csv", "baseline", "T2.2b"),
    ("resultados_condicion_A_local.csv", "condicion", "T2.3"),
    ("resultados_condicion_B_silo.csv", "condicion", "T2.4"),
    ("resultados_condicion_C_global.csv", "condicion", "T2.5"),
]


def cargar_resumen_fase2() -> pd.DataFrame:
    piezas = []
    for fichero, col_metodo, tarea in FUENTES:
        df = pd.read_csv(REPORTS / fichero)
        df = df.rename(columns={col_metodo: "metodo"})
        df["tarea"] = tarea
        piezas.append(df)
    return pd.concat(piezas, ignore_index=True)


def main() -> None:
    resumen = cargar_resumen_fase2()

    modo = os.environ.get("WANDB_MODE", "offline")
    run = wandb.init(
        project="tfm-forecasting-federado", name="fase2-resumen-final",
        mode=modo, config={"fase": "2", "n_metodos": resumen["metodo"].nunique()},
    )

    tabla = wandb.Table(dataframe=resumen)
    run.log({"resumen_fase2": tabla})

    test = resumen[resumen.split == "test"].sort_values("wmape_mediana")
    tabla_test = wandb.Table(dataframe=test[["metodo", "tarea", "wmape_mediana", "mase_mediana", "rmsse_mediana", "cobertura"]])
    run.log({
        "ranking_wmape_test": wandb.plot.bar(
            tabla_test, "metodo", "wmape_mediana",
            title="WMAPE mediana en test — los 8 métodos de la Fase 2",
        )
    })

    for _, fila in test.iterrows():
        run.summary[f"wmape_test_{fila['metodo']}"] = fila["wmape_mediana"]

    print(f"Modo W&B: {modo}")
    print(test[["metodo", "tarea", "wmape_mediana", "mase_mediana", "rmsse_mediana"]].to_string(index=False))
    run.finish()
    if modo == "offline":
        print("\nCorrida guardada en local (offline). Para subirla: "
              "`wandb login` seguido de `wandb sync wandb/latest-run` (o la carpeta que indique arriba).")


if __name__ == "__main__":
    main()
