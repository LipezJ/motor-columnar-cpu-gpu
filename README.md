# Consultas analíticas en serial, CPU paralela y GPU

Proyecto final de Programación Paralela (UIS).

**Pregunta:** ¿cuándo vale la pena ejecutar consultas analíticas en GPU frente a una CPU multinúcleo?

Implementamos a mano, en C++ y CUDA, tres consultas de TPC-H sobre datos guardados por columnas, cada una en tres versiones, y las medimos. No es un motor de SQL: cada consulta tiene su código.

| Versión | Archivo | Basado en |
|---|---|---|
| Serial: una tupla a la vez, un hilo | `src/serial.cpp` | Volcano (Graefe, 1994) |
| CPU paralela con OpenMP | `src/omp.cpp` | Morsels (Leis et al., SIGMOD 2014) |
| GPU con CUDA | `src/cuda.cu` | Crystal (Shanbhag et al., SIGMOD 2020) |

## Consultas

- **Q6**: filtro + suma. Limitada por ancho de banda de memoria.
- **Q1**: agregación en 4 grupos. Contención entre hilos.
- **Q3**: join de tres tablas. Tabla hash construida en paralelo.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install duckdb numpy matplotlib
.venv/bin/python scripts/gen_datos.py 10   # datos TPC-H y resultados esperados, con DuckDB
make                                       # make CUDA=1 CUDA_ARCH=sm_80 para incluir la GPU
./build/motor data/sf10 omp q1 --hilos 8 | .venv/bin/python scripts/validar.py data/sf10/esperado/q1.csv
.venv/bin/python scripts/bench.py --datos data/sf10 --backends serial,omp --etiqueta mac
.venv/bin/python scripts/graficas.py resultados/*.csv
```

GPU en Colab: `colab.ipynb`. Clúster SC3: `sbatch scripts/sc3_cpu.slurm` (OpenMP en smexa, 256 hilos) y `sbatch scripts/sc3_gpu.slurm` (CUDA en felix, Titan X).
