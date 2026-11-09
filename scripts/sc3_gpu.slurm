#!/bin/bash
# Backend CUDA en felix: NVIDIA GeForce GTX Titan X (Maxwell, sm_52, 12 GB).
# Uso, desde la raíz del repo: sbatch scripts/sc3_gpu.slurm
#SBATCH --job-name=motor-gpu
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# El QOS students permite como mucho 1 hora por job.
#SBATCH --time=00:30:00
#SBATCH --output=motor-bench-%j.log

# SF 10, el mismo que en Colab, para comparar la Titan X con la T4.
SF=${SF:-10}
PYTHON=${PYTHON:-python3}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# El único CUDA del clúster que sigue soportando Maxwell es el módulo cuda/11.8.
module load cuda/11.8

# CUDA 11.8 admite como mucho GCC 11, y el gcc por defecto del clúster es el 14.
# Todo el binario se compila con el g++ del sistema para no mezclar dos libstdc++ al enlazar.
GXX=/usr/bin/g++
VERSION_GXX=$($GXX -dumpversion | cut -d. -f1)
if [ "$VERSION_GXX" -gt 11 ]; then
  echo "$GXX es GCC $VERSION_GXX y CUDA 11.8 admite como mucho GCC 11; carga un gcc <= 11 y ajusta GXX" >&2
  exit 1
fi

make clean BUILD=build-gpu
make BUILD=build-gpu CUDA=1 CUDA_ARCH=sm_52 CXX="$GXX" NVCC="nvcc -ccbin $GXX"
# Datos en el /tmp del nodo: es tmpfs y persiste entre jobs.
# El home tiene cuota de 1 GB y /mnt/san/scratch no admite escritura de estudiantes.
# Se cargan una vez desde guane, con el .tar.zst que genera el Mac.
# --mem porque lo que se escribe en tmpfs cuenta como memoria del paso de SLURM.
#   srun --mem=32G -p gpu_titan -w felix sh -c 'mkdir -p /tmp/$USER && zstd -dc | tar --warning=no-unknown-keyword -xf - -C /tmp/$USER' < sf10.tar.zst
DATOS=${DATOS:-/tmp/$USER}
if [ ! -f "$DATOS/sf$SF/esperado/q6.csv" ]; then
  echo "Faltan los datos en $DATOS/sf$SF de $(hostname)" >&2
  exit 1
fi

"$PYTHON" scripts/bench.py --motor build-gpu/motor --datos "$DATOS/sf$SF" --backends cuda,cuda-ingenuo --maquina TitanX --etiqueta "TitanX-sf$SF"
