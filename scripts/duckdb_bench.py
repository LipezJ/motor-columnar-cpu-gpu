"""Mide DuckDB con las mismas consultas y el mismo SF, para comparar el motor contra uno de producción en la misma máquina.

Uso: python scripts/duckdb_bench.py SF --hilos N --maquina mac
Escribe resultados/duckdb-{maquina}-sf{SF}.csv con las columnas de scripts/bench.py y dos backends:
  duckdb           las consultas de TPC-H tal cual, sobre las tablas de DuckDB (decimales, fechas y textos);
  duckdb-enteros   las mismas consultas sobre las columnas del motor (mismos enteros, escalas y códigos que los .col).
total_ms es el tiempo de pared de la consulta completa.
computo_ms es solo la ejecución: la latencia que reporta el profiler de DuckDB menos su análisis, optimización y
planificación física, que el motor no hace (cada consulta ya está escrita como plan en C++).
Imprime además qué fracción del tiempo de ejecución se lleva cada operador de DuckDB.
"""

import argparse
import collections
import csv
import datetime
import json
import statistics
import tempfile
import time
from pathlib import Path

import duckdb

from bench import COLUMNAS as COLUMNAS_CSV
from gen_datos import COLUMNAS, ESPERADO

RAIZ = Path(__file__).resolve().parent.parent


def dia(iso: str) -> int:
    return (datetime.date.fromisoformat(iso) - datetime.date(1970, 1, 1)).days


# Mismos parámetros que src/consultas.hpp y que ESPERADO, en las escalas de los .col.
# Las sumas de DuckDB sobre BIGINT acumulan en enteros de 128 bits; el motor usa int64 porque conoce las cotas de TPC-H.
ENTEROS = {
    "q1": f"""
        SELECT l_returnflag, l_linestatus,
               sum(l_quantity::BIGINT), sum(l_extendedprice::BIGINT),
               sum(l_extendedprice::BIGINT * (100 - l_discount)),
               sum(l_extendedprice::BIGINT * (100 - l_discount) * (100 + l_tax)),
               sum(l_discount::BIGINT), count(*)
        FROM m_lineitem
        WHERE l_shipdate <= {dia('1998-12-01') - 90}
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """,
    "q3": f"""
        SELECT l_orderkey, sum(l_extendedprice::BIGINT * (100 - l_discount)) AS revenue, o_orderdate, o_shippriority
        FROM m_customer, m_orders, m_lineitem
        WHERE c_mktsegment = 1
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < {dia('1995-03-15')}
          AND l_shipdate > {dia('1995-03-15')}
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10
    """,
    "q6": f"""
        SELECT sum(l_extendedprice::BIGINT * l_discount)
        FROM m_lineitem
        WHERE l_shipdate >= {dia('1994-01-01')}
          AND l_shipdate < {dia('1995-01-01')}
          AND l_discount BETWEEN 5 AND 7
          AND l_quantity < 2400
    """,
}


def operadores(nodo: dict, acum: collections.Counter):
    """Suma operator_timing por tipo de operador en el árbol del profiler (tiempo de CPU sumado entre hilos)."""
    for hijo in nodo.get("children", []):
        if hijo.get("operator_type"):
            acum[hijo["operator_type"]] += hijo.get("operator_timing", 0)
        operadores(hijo, acum)


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
    # Las tablas m_* tienen los mismos valores y tipos que los .col (mismas expresiones de scripts/gen_datos.py).
    for tabla, cols in COLUMNAS.items():
        exprs = ", ".join(f"CAST({e} AS {'TINYINT' if tipo.__name__ == 'int8' else 'INTEGER'}) AS {n}"
                          for n, (e, tipo) in cols.items())
        con.execute(f"CREATE TABLE m_{tabla} AS SELECT {exprs} FROM {tabla}")

    perfil = Path(tempfile.mkdtemp()) / "perfil.json"
    con.execute("PRAGMA enable_profiling = 'json'")
    con.execute("SET profiling_mode = 'detailed'")
    con.execute(f"PRAGMA profiling_output = '{perfil}'")

    filas = []
    for backend, consultas in [("duckdb", ESPERADO), ("duckdb-enteros", ENTEROS)]:
        print(f"\n{backend}\nconsulta  total_ms  preparacion_ms  ejecucion_ms  operadores (% del tiempo de ejecución)")
        for q in sorted(consultas):
            for _ in range(args.calentamiento):
                con.execute(consultas[q]).fetchall()
            tiempos = []
            ops = collections.Counter()
            for rep in range(args.reps):
                ini = time.perf_counter()
                con.execute(consultas[q]).fetchall()
                ms = (time.perf_counter() - ini) * 1000
                perf = json.load(open(perfil))
                preparacion = (perf["planner"] + perf["all_optimizers"] + perf["physical_planner"]) * 1000
                ejecucion = perf["latency"] * 1000 - preparacion
                tiempos.append((ms, preparacion, ejecucion))
                if rep:
                    operadores(perf, ops)
                filas.append({"maquina": args.maquina, "sf": sf_txt, "backend": backend, "consulta": q,
                              "hilos": args.hilos, "rep": rep, "total_ms": f"{ms:.3f}", "transferencia_ms": "0.000",
                              "computo_ms": f"{ejecucion:.3f}", "valido": True})
            med = [statistics.median(x[i] for x in tiempos[1:]) for i in range(3)]
            total_ops = sum(ops.values()) or 1
            reparto = ", ".join(f"{op} {100 * t / total_ops:.0f}" for op, t in ops.most_common(4))
            print(f"{q:8}  {med[0]:8.1f}  {med[1]:14.2f}  {med[2]:12.1f}  {reparto}")

    salida = RAIZ / "resultados" / f"duckdb-{args.maquina}-sf{sf_txt}.csv"
    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        w.writeheader()
        w.writerows(filas)
    print(f"\nEscrito {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
