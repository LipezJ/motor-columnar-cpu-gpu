"""Compara la salida del motor con la de DuckDB.

Uso: build/motor data/sf1 omp q1 | python scripts/validar.py data/sf1/esperado/q1.csv
Sale con código 1 si alguna celda difiere.
"""

import csv
import math
import sys


def igual(a: str, b: str) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    # Las sumas son exactas. La tolerancia es para los promedios, que DuckDB calcula en double.
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6)


def comparar(esperado: list, obtenido: list) -> list:
    """Devuelve las diferencias entre dos tablas CSV ya leídas (filas de str, con cabecera); vacía si coinciden."""
    errores = []
    if len(esperado) != len(obtenido):
        errores.append(f"{len(obtenido) - 1} filas, se esperaban {len(esperado) - 1}")
    for i, (fe, fo) in enumerate(zip(esperado, obtenido)):
        for j, (e, o) in enumerate(zip(fe, fo)):
            if not igual(e, o):
                errores.append(f"fila {i}, columna {esperado[0][j]}: obtenido {o}, esperado {e}")
    return errores


def main():
    errores = comparar(list(csv.reader(open(sys.argv[1]))), list(csv.reader(sys.stdin)))
    if errores:
        print("\n".join(["DIFIERE"] + errores[:20]))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
