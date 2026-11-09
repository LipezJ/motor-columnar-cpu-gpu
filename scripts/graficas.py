"""Genera las gráficas del póster a partir de los CSV de scripts/bench.py.

Uso: python scripts/graficas.py resultados/*.csv [--salida resultados/graficas]

Escribe en --salida, por cada (maquina, sf):
  central_{maquina}_sf{sf}.png  mediana de serial, omp con el máximo de hilos y cuda (transferencia + cómputo)
  speedup_{maquina}_sf{sf}.png  speedup y eficiencia de omp respecto a omp con 1 hilo
y, por cada sf con cuda medido en más de una máquina:
  gpus_sf{sf}.png               cómputo de cuda por GPU
Descarta la rep 0 (calentamiento) y las filas con valido=False.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

# Okabe-Ito (distinguible con daltonismo y en impresión en escala de grises).
COLOR = {"serial": "#999999", "omp": "#0072B2", "cuda": "#009E73", "transferencia": "#A8DCC9"}
# Por encima de esta razón entre la barra más alta y la más baja, el eje y pasa a escala log.
RAZON_LOG = 30

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 16, "legend.fontsize": 16,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def leer(rutas):
    """Medianas por (maquina, sf, backend, consulta, hilos) -> {"total", "transferencia", "computo"} en ms."""
    medidas = defaultdict(lambda: defaultdict(list))
    for ruta in rutas:
        for r in csv.DictReader(open(ruta)):
            if r["valido"] != "True" or r["rep"] in ("", "0"):
                continue
            clave = (r["maquina"], r["sf"], r["backend"], r["consulta"], int(r["hilos"]))
            for campo in ("total", "transferencia", "computo"):
                medidas[clave][campo].append(float(r[campo + "_ms"]))
    return {k: {c: statistics.median(v) for c, v in m.items()} for k, m in medidas.items()}


def ordenar_consultas(consultas):
    return sorted(consultas, key=lambda q: int(q[1:]))


def etiquetar(ax, barras, valores):
    for b, v in zip(barras, valores):
        if v:
            ax.annotate(f"{v:.0f}" if v >= 100 else f"{v:.3g}", (b.get_x() + b.get_width() / 2, b.get_y() + b.get_height()),
                        ha="center", va="bottom", fontsize=13, xytext=(0, 2), textcoords="offset points")


def central(med, maquina, sf, salida):
    datos = {k: v for k, v in med.items() if k[:2] == (maquina, sf)}
    consultas = ordenar_consultas({k[3] for k in datos})
    max_hilos = max((k[4] for k in datos if k[2] == "omp"), default=None)
    series = [("serial", 1, "Serial"), ("omp", max_hilos, f"OpenMP ({max_hilos} hilos)"), ("cuda", 1, "CUDA")]
    series = [s for s in series if any(k[2] == s[0] and k[4] == s[1] for k in datos)]
    if not series:
        return
    ancho = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(10, 6))
    alturas = []
    for i, (backend, hilos, nombre) in enumerate(series):
        xs = [j + (i - (len(series) - 1) / 2) * ancho for j in range(len(consultas))]
        filas = [datos.get((maquina, sf, backend, q, hilos)) for q in consultas]
        if backend == "cuda":
            transf = [f["transferencia"] if f else 0 for f in filas]
            comp = [f["computo"] if f else 0 for f in filas]
            ax.bar(xs, transf, ancho, color=COLOR["transferencia"], label="CUDA: transferencia")
            barras = ax.bar(xs, comp, ancho, bottom=transf, color=COLOR["cuda"], label="CUDA: cómputo")
            totales = [t + c for t, c in zip(transf, comp)]
        else:
            totales = [f["total"] if f else 0 for f in filas]
            barras = ax.bar(xs, totales, ancho, color=COLOR[backend], label=nombre)
        etiquetar(ax, barras, totales)
        alturas += [t for t in totales if t]
    if max(alturas) / min(alturas) > RAZON_LOG:
        ax.set_yscale("log")
    ax.set_xticks(range(len(consultas)))
    ax.set_xticklabels([q.upper() for q in consultas])
    ax.set_ylabel("Tiempo (ms, mediana)")
    ax.legend(frameon=False)
    guardar(fig, salida / f"central_{maquina}_sf{sf}.png")


def speedup(med, maquina, sf, salida):
    omp = {k: v for k, v in med.items() if k[:3] == (maquina, sf, "omp")}
    consultas = [q for q in ordenar_consultas({k[3] for k in omp}) if (maquina, sf, "omp", q, 1) in omp]
    if not consultas:
        return
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 6))
    max_hilos = 1
    for q in consultas:
        hilos = sorted(k[4] for k in omp if k[3] == q)
        base = omp[(maquina, sf, "omp", q, 1)]["total"]
        s = [base / omp[(maquina, sf, "omp", q, h)]["total"] for h in hilos]
        ax_s.plot(hilos, s, "o-", lw=2.5, ms=8, label=q.upper())
        ax_e.plot(hilos, [v / h for v, h in zip(s, hilos)], "o-", lw=2.5, ms=8, label=q.upper())
        max_hilos = max(max_hilos, hilos[-1])
    ax_s.plot([1, max_hilos], [1, max_hilos], "--", color="black", lw=1.5, label="Ideal")
    ax_e.axhline(1, ls="--", color="black", lw=1.5)
    for ax in (ax_s, ax_e):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({k[4] for k in omp}))
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Hilos")
    ax_s.set_ylabel("Speedup vs. 1 hilo")
    ax_e.set_ylabel("Eficiencia")
    ax_e.set_ylim(0, 1.1)
    ax_s.legend(frameon=False)
    guardar(fig, salida / f"speedup_{maquina}_sf{sf}.png")


def gpus(med, sf, salida):
    cuda = {k: v for k, v in med.items() if k[1] == sf and k[2] == "cuda"}
    maquinas = sorted({k[0] for k in cuda})
    if len(maquinas) < 2:
        return
    consultas = ordenar_consultas({k[3] for k in cuda})
    ancho = 0.8 / len(maquinas)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(maquinas):
        xs = [j + (i - (len(maquinas) - 1) / 2) * ancho for j in range(len(consultas))]
        valores = [cuda[(m, sf, "cuda", q, 1)]["computo"] if (m, sf, "cuda", q, 1) in cuda else 0 for q in consultas]
        etiquetar(ax, ax.bar(xs, valores, ancho, label=m), valores)
    ax.set_xticks(range(len(consultas)))
    ax.set_xticklabels([q.upper() for q in consultas])
    ax.set_ylabel("Cómputo en GPU (ms, mediana)")
    ax.legend(frameon=False)
    guardar(fig, salida / f"gpus_sf{sf}.png")


def guardar(fig, ruta):
    fig.savefig(ruta)
    plt.close(fig)
    print(ruta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--salida", type=Path, default=Path("resultados/graficas"))
    args = p.parse_args()
    med = leer(args.csv)
    if not med:
        sys.exit("No hay filas válidas (valido=True, rep > 0) en los CSV.")
    args.salida.mkdir(parents=True, exist_ok=True)
    for maquina, sf in sorted({k[:2] for k in med}):
        central(med, maquina, sf, args.salida)
        speedup(med, maquina, sf, args.salida)
    for sf in sorted({k[1] for k in med}):
        gpus(med, sf, args.salida)


if __name__ == "__main__":
    main()
