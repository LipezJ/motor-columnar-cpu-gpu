"""Mide DuckDB con las mismas consultas y el mismo SF, para comparar el motor contra uno de producción en la misma máquina.

Uso: python scripts/duckdb_bench.py SF --hilos N --maquina mac
Escribe resultados/duckdb-{maquina}-sf{SF}.csv con las columnas de scripts/bench.py (backend = duckdb).
total_ms es el tiempo de pared de la consulta completa.
computo_ms es solo la ejecución: la latencia que reporta el profiler de DuckDB menos su análisis, optimización y
planificación física, que nuestro motor no hace (cada consulta ya está escrita como plan en C++).
"""

import argparse
import csv
import json
import tempfile
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
    p.add_argument("--reps", type=int, default=6, help="repeticiones medidas por consulta, contando la rep 0 que descarta graficas.py")
    # DuckDB tarda varias ejecuciones en estabilizarse (en el Mac, Q3 pasa de ~250 ms a ~55 ms hacia la séptima).
    # El motor no lo necesita: sus tiempos ya son estables desde la segunda repetición.
    p.add_argument("--calentamiento", type=int, default=10, help="ejecuciones previas por consulta que no se registran")
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

    perfil = Path(tempfile.mkdtemp()) / "perfil.json"
    con.execute("PRAGMA enable_profiling = 'json'")
    con.execute("SET profiling_mode = 'detailed'")
    con.execute(f"PRAGMA profiling_output = '{perfil}'")

    filas = []
    print("consulta  total_ms  preparacion_ms  ejecucion_ms")
    for q, sql in ESPERADO.items():
        for _ in range(args.calentamiento):
            con.execute(sql).fetchall()
        tiempos = []
        for rep in range(args.reps):
            ini = time.perf_counter()
            con.execute(sql).fetchall()
            ms = (time.perf_counter() - ini) * 1000
            p = json.load(open(perfil))
            preparacion = (p["planner"] + p["all_optimizers"] + p["physical_planner"]) * 1000
            ejecucion = p["latency"] * 1000 - preparacion
            tiempos.append((ms, preparacion, ejecucion))
            filas.append({"maquina": args.maquina, "sf": sf_txt, "backend": "duckdb", "consulta": q, "hilos": args.hilos,
                          "rep": rep, "total_ms": f"{ms:.3f}", "transferencia_ms": "0.000",
                          "computo_ms": f"{ejecucion:.3f}", "valido": True})
        med = [statistics.median(x[i] for x in tiempos[1:]) for i in range(3)]
        print(f"{q:8}  {med[0]:8.1f}  {med[1]:14.2f}  {med[2]:12.1f}")

    salida = RAIZ / "resultados" / f"duckdb-{args.maquina}-sf{sf_txt}.csv"
    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    print(f"Escrito {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
