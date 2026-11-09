#!/bin/bash
# Uso (desde la raíz del repo): sbatch scripts/sc3.slurm
#
# Rellenar antes de enviar (los nombres salen de `sinfo -o "%P %c %m %G"`):
#   --partition        partición del nodo (una con GPU si se quiere medir cuda)
#   --cpus-per-task    núcleos del nodo; bench.py barre potencias de 2 hasta los núcleos asignados
#   --gres             GPU pedida, p. ej. gpu:1 o gpu:a100:1; borrar la línea en una partición sin GPU
# SLURM lee las líneas #SBATCH antes de ejecutar el script, así que no pueden usar variables de shell.
#SBATCH --job-name=motor-bench
#SBATCH --partition=RELLENAR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=RELLENAR
#SBATCH --gres=gpu:1
# Nodo exclusivo: otro trabajo en el nodo compartiría el ancho de banda de memoria y falsearía el speedup.
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

# Factor de escala de TPC-H.
SF=10
# Python con duckdb, numpy y matplotlib (los nodos de cómputo pueden no tener internet: crear el venv en el nodo de login).
PYTHON=${PYTHON:-.venv/bin/python}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Sin fijar hilos a núcleos, el planificador del SO los migra y el speedup de omp sale ruidoso.
export OMP_PROC_BIND=close OMP_PLACES=cores

make clean
BACKENDS=serial,omp
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  make CUDA=1 CUDA_ARCH="$ARCH"
  BACKENDS=serial,omp,cuda
else
  make
fi

[ -d "data/sf$SF" ] || "$PYTHON" scripts/gen_datos.py "$SF"

"$PYTHON" scripts/bench.py --datos "data/sf$SF" --backends "$BACKENDS" --etiqueta "sc3-sf$SF" --maquina sc3
"$PYTHON" scripts/graficas.py resultados/*.csv
