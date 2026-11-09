"""Mide DuckDB con las mismas consultas y el mismo SF, para comparar el motor contra uno de producción en la misma máquina.

Uso: python scripts/duckdb_bench.py SF --hilos N --maquina mac
Escribe resultados/duckdb-{maquina}-sf{SF}.csv con las columnas de scripts/bench.py (backend = duckdb).
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import duckdb

from bench import COLUMNAS
from gen_datos import ESPERADO

RAIZ = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sf", type=float)
    p.add_argument("--hilos", type=int, required=True)
    p.add_argument("--reps", type=int, default=5, help="repeticiones por consulta, contando la rep 0 de calentamiento")
    p.add_argument("--maquina", required=True)
    args = p.parse_args()

    sf_txt = f"{args.sf:g}"
    # Base en memoria: DuckDB lee de su propio formato columnar en RAM, igual que el motor lee sus columnas ya cargadas.
    # La generación queda fuera de lo medido.
    con = duckdb.connect()
    con.execute(f"SET threads = {args.hilos}")
    con.execute("INSTALL tpch; LOAD tpch;")
    print(f"Generando TPC-H SF={sf_txt} en memoria")
    con.execute(f"CALL dbgen(sf={args.sf})")

    filas = []
    print("consulta  mediana_ms")
    for q, sql in ESPERADO.items():
        tiempos = []
        for rep in range(args.reps):
            ini = time.perf_counter()
            con.execute(sql).fetchall()
            ms = (time.perf_counter() - ini) * 1000
            tiempos.append(ms)
            filas.append({"maquina": args.maquina, "sf": sf_txt, "backend": "duckdb", "consulta": q, "hilos": args.hilos,
                          "rep": rep, "total_ms": f"{ms:.3f}", "transferencia_ms": "0.000",
                          "computo_ms": f"{ms:.3f}", "valido": True})
        print(f"{q:8}  {statistics.median(tiempos[1:]):10.3f}")

    salida = RAIZ / "resultados" / f"duckdb-{args.maquina}-sf{sf_txt}.csv"
    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    print(f"Escrito {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
