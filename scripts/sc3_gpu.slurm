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
#SBATCH --time=02:00:00
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
test -f "data/sf$SF/esperado/q6.csv" || "$PYTHON" scripts/gen_datos.py "$SF"

"$PYTHON" scripts/bench.py --motor build-gpu/motor --datos "data/sf$SF" --backends cuda --maquina TitanX --etiqueta "TitanX-sf$SF"
