"""calendario_semanal.py — utilidad compartida por T1.4 y T2.2.

Reconstruye, por serie (tienda x familia), el calendario semanal COMPLETO entre su primera
y su ultima semana presente -- incluyendo semanas excluidas en T1.3 como "parciales" (la mas
notable: la semana de Navidad, ausente en las 4 navidades del dataset porque el 25-dic no
tiene NINGUNA fila en train.csv para ninguna tienda/familia -- tiendas cerradas ese dia).

Por que hace falta: groupby().shift(N) avanza por POSICION dentro del grupo, no por fecha. Si
una serie tiene un hueco interno (una semana ausente), shift(N) tras el hueco deja de
corresponder a "N semanas atras" de verdad -- silenciosamente coge una semana mas antigua de
lo que su nombre indica. Reindexar a un calendario completo (huecos = NaN) antes de hacer
shift/rolling hace que la aritmetica posicional vuelva a coincidir con la aritmetica de
calendario: shift(N) tras un hueco da NaN (correcto) en vez de un valor de la semana
equivocada (silenciosamente incorrecto).
"""
import pandas as pd


def construir_calendario_completo(
    data: pd.DataFrame,
    cols_valor: list[str],
    id_cols: tuple[str, ...] = ("store_nbr", "family"),
) -> pd.DataFrame:
    """Devuelve una copia de `data` reindexada a semanas consecutivas (freq=7D) por serie,
    con NaN en cols_valor para cualquier semana ausente en el original."""
    piezas = []
    for claves, g in data.groupby(list(id_cols), sort=False):
        claves = claves if isinstance(claves, tuple) else (claves,)
        semanas = pd.date_range(g["week_start"].min(), g["week_start"].max(), freq="7D")
        fila_id = {col: val for col, val in zip(id_cols, claves)}
        piezas.append(pd.DataFrame({**fila_id, "week_start": semanas}))
    calendario = pd.concat(piezas, ignore_index=True)
    completo = calendario.merge(
        data[list(id_cols) + ["week_start"] + cols_valor],
        on=list(id_cols) + ["week_start"],
        how="left",
    )
    return completo.sort_values(list(id_cols) + ["week_start"]).reset_index(drop=True)
