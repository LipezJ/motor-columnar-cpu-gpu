#!/bin/bash
# Curva de speedup de OpenMP en smexa: 2 AMD EPYC 9534 (2 x 64 núcleos, 256 hilos, 2 nodos NUMA).
# Uso, desde la raíz del repo: sbatch scripts/sc3_cpu.slurm
#SBATCH --job-name=motor-cpu
#SBATCH --partition=amd
#SBATCH --nodelist=smexa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
# Nodo exclusivo: otro trabajo en el nodo compartiría el ancho de banda de memoria y falsearía el speedup.
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=motor-bench-%j.log

# SF 50 y no 10: con 256 hilos, SF 10 deja ~3 morsels por hilo y el desbalance tapa el speedup.
# Por encima de SF ~80 desborda sum_charge de Q1 (ver FilaQ1 en src/consultas.hpp).
SF=${SF:-50}
PYTHON=${PYTHON:-python3}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# close + cores: los primeros 64 hilos caen en el socket 0, que es donde quedan los datos (los carga un solo hilo).
# Así la curva muestra el salto al pasar de 64 a 128 hilos, cuando empiezan los accesos al otro nodo NUMA.
export OMP_PROC_BIND=close OMP_PLACES=cores

make clean BUILD=build-cpu
make BUILD=build-cpu
# Datos en el /tmp del nodo: es tmpfs y persiste entre jobs.
# El home tiene cuota de 1 GB y /mnt/san/scratch no admite escritura de estudiantes.
# Se cargan una vez desde guane, con el .tar.zst que genera el Mac:
#   srun -p amd -w smexa sh -c 'mkdir -p /tmp/$USER && zstd -dc | tar -xf - -C /tmp/$USER' < sf50.tar.zst
DATOS=${DATOS:-/tmp/$USER}
if [ ! -f "$DATOS/sf$SF/esperado/q6.csv" ]; then
  echo "Faltan los datos en $DATOS/sf$SF de $(hostname)" >&2
  exit 1
fi

"$PYTHON" scripts/bench.py --motor build-cpu/motor --datos "$DATOS/sf$SF" --backends serial,omp --maquina smexa --etiqueta "smexa-sf$SF"

# Misma corrida con las páginas repartidas entre los dos nodos NUMA (numactl hereda la política al motor).
# Compara contra la anterior el coste de que todos los datos vivan en un solo socket.
if command -v numactl >/dev/null; then
  numactl --interleave=all "$PYTHON" scripts/bench.py --motor build-cpu/motor --datos "$DATOS/sf$SF" --backends omp \
    --maquina smexa-interleave --etiqueta "smexa-interleave-sf$SF"
else
  echo "numactl no está instalado en $(hostname): se omite la corrida con memoria intercalada"
fi
