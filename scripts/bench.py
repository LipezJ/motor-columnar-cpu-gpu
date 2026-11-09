"""Corre build/motor sobre una matriz de configuraciones y guarda los tiempos en resultados/{etiqueta}.csv.

Uso: python scripts/bench.py --datos data/sf10 --backends serial,omp,cuda --etiqueta sc3

Cada invocación del motor mide --reps repeticiones; todas van al CSV.
La rep 0 es de calentamiento: se guarda, pero el resumen (y scripts/graficas.py) la descartan.
Una ejecución que falla deja una fila con rep vacía y valido=False, y la corrida sigue.
"""

import argparse
import csv
import io
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validar import comparar  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
COLUMNAS = ["maquina", "sf", "backend", "consulta", "hilos", "rep", "total_ms", "transferencia_ms", "computo_ms", "valido"]


def nombre_gpu() -> str:
    """Nombre corto de la primera GPU ("T4", "L4", "A100-SXM4-40GB"), o "" si no hay nvidia-smi."""
    try:
        salida = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""
    nombre = salida.splitlines()[0].strip()
    return re.sub(r"^(NVIDIA|Tesla)\s+", "", nombre).replace(" ", "-")


def hilos_por_defecto() -> str:
    # sched_getaffinity respeta los núcleos que SLURM asigna al trabajo; os.cpu_count() cuenta los del nodo entero.
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    potencias = [2 ** k for k in range(n.bit_length()) if 2 ** k < n]
    return ",".join(map(str, potencias + [n]))


def correr(motor, datos, backend, consulta, hilos, reps):
    """Devuelve (filas de tiempos como dicts, errores); errores vacío si la salida coincide con DuckDB."""
    cmd = [str(motor), str(datos), backend, consulta, "--reps", str(reps)]
    if backend == "omp":
        cmd += ["--hilos", str(hilos)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lineas = p.stderr.splitlines()
    # main.cpp mezcla en stderr los tiempos en CSV con los mensajes de error; solo las filas numéricas son tiempos.
    tiempos = [l for l in lineas if re.fullmatch(r"\d+,\d+,[\d.]+,[\d.]+,[\d.]+", l)]
    mensajes = [l for l in lineas if l not in tiempos and not l.startswith("rep,")]
    if p.returncode != 0:
        return [], [f"código {p.returncode}"] + mensajes
    filas = list(csv.DictReader(io.StringIO("rep,hilos,total_ms,transferencia_ms,computo_ms\n" + "\n".join(tiempos))))
    esperado = list(csv.reader(open(Path(datos) / "esperado" / f"{consulta}.csv")))
    return filas, comparar(esperado, list(csv.reader(io.StringIO(p.stdout))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default="data/sf10", help="directorio de gen_datos.py (data/sfN)")
    p.add_argument("--backends", default="serial,omp,cuda")
    p.add_argument("--consultas", default="q1,q3,q6")
    p.add_argument("--hilos", default=hilos_por_defecto(),
                   help="lista de hilos para omp; serial y cuda corren siempre con 1 (por defecto, potencias de 2 hasta los núcleos disponibles)")
    p.add_argument("--reps", type=int, default=5, help="repeticiones por configuración, contando la rep 0 de calentamiento")
    p.add_argument("--maquina", default=nombre_gpu() or platform.node().split(".")[0],
                   help="valor de la columna maquina (por defecto, el nombre de la GPU o el del host)")
    p.add_argument("--etiqueta", help="nombre del CSV en resultados/ (por defecto, --maquina)")
    p.add_argument("--motor", default=str(RAIZ / "build" / "motor"))
    args = p.parse_args()

    sf = re.sub(r"^sf", "", Path(args.datos).name)
    salida = RAIZ / "resultados" / f"{args.etiqueta or args.maquina}.csv"
    salida.parent.mkdir(exist_ok=True)
    medianas = []
    fallos = 0

    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNAS)
        w.writeheader()
        for backend in args.backends.split(","):
            for consulta in args.consultas.split(","):
                for hilos in (args.hilos.split(",") if backend == "omp" else ["1"]):
                    filas, errores = correr(args.motor, args.datos, backend, consulta, hilos, args.reps)
                    base = {"maquina": args.maquina, "sf": sf, "backend": backend, "consulta": consulta,
                            "hilos": hilos, "valido": not errores}
                    if errores:
                        fallos += 1
                        print(f"[{backend} {consulta} {hilos}h] " + "\n  ".join(errores[:20]), file=sys.stderr)
                    if not filas:
                        w.writerow(base)
                    for r in filas:
                        w.writerow({**base, **r})
                    f.flush()
                    medidas = [float(r["total_ms"]) for r in filas if r["rep"] != "0"]
                    if medidas:
                        medianas.append((backend, consulta, hilos, statistics.median(medidas), not errores))

    print(f"{'backend':8} {'consulta':8} {'hilos':>5} {'mediana_ms':>12}  válido")
    for backend, consulta, hilos, m, ok in medianas:
        print(f"{backend:8} {consulta:8} {hilos:>5} {m:12.3f}  {'sí' if ok else 'NO'}")
    print(f"Escrito {salida.relative_to(RAIZ)} ({fallos} configuraciones fallidas)")


if __name__ == "__main__":
    main()
