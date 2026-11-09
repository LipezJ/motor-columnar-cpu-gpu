"""Genera las figuras del póster a partir de los CSV de resultados/.

Uso: python scripts/graficas.py [--resultados resultados] [--salida resultados/graficas]

Escribe cada figura en PDF (la que incluye poster/poster.tex) y en PNG:
  central     filas por segundo por consulta: 1 núcleo, CPU completa, GPU con datos residentes y GPU copiando por PCIe
  escalado    speedup de OpenMP y ancho de banda alcanzado en smexa, con datos en un socket y con memoria intercalada
  contencion  Q1 en GPU: agregación por bloque contra un atómico global por fila, en cada GPU
Usa la mediana de cada configuración, sin la rep 0 (calentamiento) y sin filas con valido=False.
Una figura cuyos datos no están se omite con un aviso.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent

# Filas de lineitem que imprime scripts/gen_datos.py para cada SF.
FILAS_LINEITEM = {"10": 59_986_052, "50": 300_005_811}
# Bytes de lineitem que lee cada consulta por fila en src/omp.cpp (columnas int32 de 4 B e int8 de 1 B).
# Q6 lee 4 columnas int32 en todas las filas (filtro sin ramas).
# Q1 lee 5 columnas int32 y 2 int8 (el filtro de fecha deja pasar el ~98 %).
# Q3 no aparece: sus accesos a la tabla hash son aleatorios y no se miden en ancho de banda.
BYTES_POR_FILA = {"q6": 16, "q1": 22}
# Pico teórico de smexa: 12 canales DDR5-4800 por socket (8 B por transferencia), 2 sockets.
PICO_SOCKET_GBS = 12 * 4.8 * 8
NUCLEOS_SOCKET = 64

# Okabe-Ito (distinguible con daltonismo y en escala de grises).
COLOR = {"q1": "#0072B2", "q3": "#E69F00", "q6": "#009E73",
         "serial": "#999999", "cpu": "#0072B2", "gpu": "#009E73", "pcie": "#A8DCC9", "ingenuo": "#D55E00"}
NOMBRE = {"q1": "Q1 (agregación)", "q3": "Q3 (join)", "q6": "Q6 (filtro + suma)"}
CONSULTAS = ["q6", "q1", "q3"]

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 16, "legend.fontsize": 15,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def medianas(directorio: Path) -> dict:
    """(maquina, backend, consulta, hilos) -> (total_ms, transferencia_ms, computo_ms), medianas sin la rep 0."""
    medidas = defaultdict(list)
    for ruta in sorted(directorio.glob("*.csv")):
        for r in csv.DictReader(open(ruta)):
            if r["valido"] != "True" or r["rep"] in ("", "0"):
                continue
            clave = (r["maquina"], r["backend"], r["consulta"], int(r["hilos"]))
            medidas[clave].append((float(r["total_ms"]), float(r["transferencia_ms"]), float(r["computo_ms"])))
    return {k: tuple(statistics.median(m[i] for m in v) for i in range(3)) for k, v in medidas.items()}


def guardar(fig, salida: Path, nombre: str):
    for ext in ("pdf", "png"):
        fig.savefig(salida / f"{nombre}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  {nombre}")


def central(m: dict, salida: Path):
    # La CPU se midió en SF 50 y la GPU en SF 10, así que se comparan filas por segundo y no milisegundos.
    def mejor_cpu(q):
        tiempos = [v[0] for (maq, b, c, _), v in m.items() if maq.startswith("smexa") and b == "omp" and c == q]
        return min(tiempos) if tiempos else None

    series = [
        ("1 núcleo CPU (serial)", COLOR["serial"], lambda q: m.get(("smexa", "serial", q, 1), (None,))[0], "50"),
        ("2 AMD EPYC, 128 núcleos (OpenMP)", COLOR["cpu"], mejor_cpu, "50"),
        ("GPU T4, datos ya en la GPU", COLOR["gpu"], lambda q: m.get(("T4", "cuda", q, 1), (0, 0, None))[2], "10"),
        ("GPU T4, copiando por PCIe", COLOR["pcie"], lambda q: m.get(("T4", "cuda", q, 1), (None,))[0], "10"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ancho = 0.2
    for i, (nombre, color, tiempo, sf) in enumerate(series):
        for j, q in enumerate(CONSULTAS):
            ms = tiempo(q)
            if not ms:
                continue
            gfilas = FILAS_LINEITEM[sf] / (ms / 1000) / 1e9
            x = j + (i - 1.5) * ancho
            ax.bar(x, gfilas, ancho, color=color, label=nombre if j == 0 else None)
            ax.text(x, gfilas * 1.08, f"{gfilas:.2g}", ha="center", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylabel("Miles de millones de filas/s")
    ax.set_xticks(range(len(CONSULTAS)), [NOMBRE[q] for q in CONSULTAS])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    guardar(fig, salida, "central")


def escalado(m: dict, salida: Path):
    corridas = [("smexa", "datos en un socket", "--"), ("smexa-interleave", "memoria intercalada", "-")]
    if not any(k[0] == "smexa" and k[1] == "omp" for k in m):
        print("  (sin datos de smexa: se omite escalado)")
        return
    fig, (izq, der) = plt.subplots(1, 2, figsize=(16, 6.5))
    for maq, etiqueta, estilo in corridas:
        for q in CONSULTAS:
            puntos = sorted((h, v[0]) for (mq, b, c, h), v in m.items() if mq == maq and b == "omp" and c == q)
            if not puntos:
                continue
            hilos = [h for h, _ in puntos]
            base = dict(puntos)[1]
            izq.plot(hilos, [base / t for _, t in puntos], estilo, marker="o", color=COLOR[q], lw=2.5)
            if q in BYTES_POR_FILA:
                gbs = [FILAS_LINEITEM["50"] * BYTES_POR_FILA[q] / (t / 1000) / 1e9 for _, t in puntos]
                der.plot(hilos, gbs, estilo, marker="o", color=COLOR[q], lw=2.5)
    tope = 2 * NUCLEOS_SOCKET
    izq.plot([1, tope], [1, tope], ":", color="black", lw=1.5)
    for ax in (izq, der):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Hilos")
        ax.axvline(NUCLEOS_SOCKET, color="#bbbbbb", lw=1)
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    izq.set_yscale("log", base=2)
    izq.set_yticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
    izq.set_ylabel("Speedup respecto a 1 hilo")
    der.set_yscale("log", base=2)
    der.set_yticks([16, 32, 64, 128, 256, 512, 1024], ["16", "32", "64", "128", "256", "512", "1024"])
    der.set_ylabel("Ancho de banda leído (GB/s)")
    # Una sola leyenda: el color es la consulta y el trazo la colocación de la memoria.
    leyenda = [Line2D([], [], color=COLOR[q], lw=3, label=NOMBRE[q]) for q in CONSULTAS]
    leyenda += [Line2D([], [], color="black", ls=e, lw=2, label=et) for _, et, e in corridas]
    leyenda.append(Line2D([], [], color="black", ls=":", lw=1.5, label="ideal (1 hilo por núcleo)"))
    fig.legend(handles=leyenda, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    for gbs, texto in [(PICO_SOCKET_GBS, "pico de 1 socket"), (2 * PICO_SOCKET_GBS, "pico de 2 sockets")]:
        der.axhline(gbs, color="black", ls=":", lw=1.5)
        der.text(1, gbs * 1.05, texto, fontsize=13)
    guardar(fig, salida, "escalado")


def contencion(m: dict, salida: Path):
    gpus = sorted({maq for (maq, b, c, _) in m if b == "cuda-ingenuo" and c == "q1"})
    if not gpus:
        print("  (sin cuda-ingenuo: se omite contencion)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ancho = 0.35
    for i, (backend, nombre, color) in enumerate([("cuda", "Agregación por bloque (memoria compartida)", COLOR["gpu"]),
                                                  ("cuda-ingenuo", "Un atómico global por fila", COLOR["ingenuo"])]):
        for j, gpu in enumerate(gpus):
            ms = m[(gpu, backend, "q1", 1)][2]
            x = j + (i - 0.5) * ancho
            ax.bar(x, ms, ancho, color=color, label=nombre if j == 0 else None)
            ax.text(x, ms * 1.02, f"{ms:.0f}", ha="center", fontsize=14)
    ax.set_xticks(range(len(gpus)), gpus)
    ax.set_ylabel("Cómputo de Q1 en GPU (ms, SF 10)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=13)
    guardar(fig, salida, "contencion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resultados", default=str(RAIZ / "resultados"))
    p.add_argument("--salida", default=str(RAIZ / "resultados" / "graficas"))
    args = p.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    m = medianas(Path(args.resultados))
    print(f"Figuras en {salida}:")
    central(m, salida)
    escalado(m, salida)
    contencion(m, salida)


if __name__ == "__main__":
    main()
