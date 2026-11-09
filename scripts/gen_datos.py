"""Genera TPC-H con DuckDB, lo exporta a columnas binarias y guarda los resultados esperados.

Uso: python scripts/gen_datos.py SF [--salida data]

Escribe data/sf{SF}/{tabla}/{columna}.col (arreglos crudos little-endian, sin cabecera)
y data/sf{SF}/esperado/q{1,3,6}.csv con el formato que imprime el motor.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np

# Los decimales se guardan en centavos (escala 100) como int32.
# El precio máximo de TPC-H (~104 950,00) cabe de sobra en int32 en centavos.
# Las fechas se guardan como días desde 1970-01-01.
# Las claves caben en int32 hasta SF ~350 (o_orderkey llega a 6 000 000 · SF).
COLUMNAS = {
    "lineitem": {
        "l_orderkey": ("l_orderkey", np.int32),
        "l_quantity": ("CAST(l_quantity * 100 AS INTEGER)", np.int32),
        "l_extendedprice": ("CAST(l_extendedprice * 100 AS INTEGER)", np.int32),
        "l_discount": ("CAST(l_discount * 100 AS INTEGER)", np.int32),
        "l_tax": ("CAST(l_tax * 100 AS INTEGER)", np.int32),
        "l_shipdate": ("l_shipdate - DATE '1970-01-01'", np.int32),
        "l_returnflag": ("ascii(l_returnflag)", np.int8),
        "l_linestatus": ("ascii(l_linestatus)", np.int8),
    },
    "orders": {
        "o_orderkey": ("o_orderkey", np.int32),
        "o_custkey": ("o_custkey", np.int32),
        "o_orderdate": ("o_orderdate - DATE '1970-01-01'", np.int32),
        "o_shippriority": ("o_shippriority", np.int32),
    },
    "customer": {
        "c_custkey": ("c_custkey", np.int32),
        # El código de cada segmento es su posición en SEGMENTOS.
        # Tiene que coincidir con SEGMENTO_BUILDING en src/consultas.hpp.
        "c_mktsegment": ("list_position(['AUTOMOBILE','BUILDING','FURNITURE','HOUSEHOLD','MACHINERY'], c_mktsegment) - 1", np.int8),
    },
}

# Mismas columnas y en el mismo orden que imprime src/resultados.cpp (scripts/validar.py compara por posición).
ESPERADO = {
    "q1": """
        SELECT l_returnflag, l_linestatus,
               sum(l_quantity) AS sum_qty,
               sum(l_extendedprice) AS sum_base_price,
               sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
               sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
               avg(l_quantity) AS avg_qty,
               avg(l_extendedprice) AS avg_price,
               avg(l_discount) AS avg_disc,
               count(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL 90 DAY
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """,
    "q3": """
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate, o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < DATE '1995-03-15'
          AND l_shipdate > DATE '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10
    """,
    "q6": """
        SELECT sum(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate < DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24
    """,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sf", type=float, help="factor de escala de TPC-H (1 ≈ 6 M filas de lineitem)")
    p.add_argument("--salida", default="data")
    args = p.parse_args()

    sf_txt = f"{args.sf:g}"
    raiz = Path(args.salida) / f"sf{sf_txt}"
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    print(f"Generando TPC-H SF={sf_txt}…")
    con.execute(f"CALL dbgen(sf={args.sf})")

    for tabla, cols in COLUMNAS.items():
        (raiz / tabla).mkdir(parents=True, exist_ok=True)
        for nombre, (expr, tipo) in cols.items():
            arr = con.execute(f"SELECT {expr} AS v FROM {tabla}").fetchnumpy()["v"]
            arr.astype(tipo).tofile(raiz / tabla / f"{nombre}.col")
        print(f"  {tabla}: {len(arr):,} filas")

    (raiz / "esperado").mkdir(exist_ok=True)
    for q, sql in ESPERADO.items():
        con.execute(f"COPY ({sql}) TO '{raiz / 'esperado' / (q + '.csv')}' (HEADER)")
    print(f"Listo: {raiz}")


if __name__ == "__main__":
    main()
