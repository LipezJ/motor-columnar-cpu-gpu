# Motor de consultas analíticas columnar: serial vs CPU paralela vs GPU

Proyecto final de Programación Paralela (UIS).

**Pregunta:** ¿cuándo vale la pena ejecutar consultas analíticas en GPU frente a una CPU multinúcleo?

Un mismo motor columnar con tres backends, evaluado con consultas de TPC-H:

| Backend | Modelo | Referencia |
|---|---|---|
| Serial | Volcano, una tupla a la vez, un hilo | SQLite clásico |
| CPU paralela | Paralelismo por *morsels* con OpenMP, hash join y agregación en paralelo | DuckDB / HyPer (Leis et al., SIGMOD 2014) |
| GPU | Kernels CUDA por *tiles*, hash join con `atomicCAS` | Crystal (Shanbhag et al., SIGMOD 2020) |

## Consultas

- **Q6**: filtro + suma; limitada por ancho de banda de memoria.
- **Q1**: agregación con 4 grupos; contención de atómicos en GPU.
- **Q3**: join de tres tablas; construcción de la tabla hash en paralelo.

## Datos

Generados con DuckDB (`CALL dbgen(sf=N)`) y exportados a columnas binarias en `data/` (no versionado).
Los resultados de cada consulta se validan contra DuckDB.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install duckdb numpy matplotlib
.venv/bin/python scripts/gen_datos.py 10
make                      # serial + omp; make CUDA=1 CUDA_ARCH=sm_80 para añadir cuda
./build/motor data/sf10 omp q1 --hilos 8 | .venv/bin/python scripts/validar.py data/sf10/esperado/q1.csv
.venv/bin/python scripts/bench.py --datos data/sf10 --backends serial,omp --etiqueta mac
.venv/bin/python scripts/graficas.py resultados/*.csv
```

En Colab: `colab.ipynb`. En el clúster SC3: rellenar la partición en `scripts/sc3.slurm` y `sbatch scripts/sc3.slurm`.

## Entornos

- **Colab Pro**: desarrollo de kernels CUDA; comparación T4 / L4 / A100.
- **Clúster SC3 (UIS)**: mediciones finales, CPU de 1 a N núcleos y GPU.
- **Comparación externa**: DuckDB (CPU), cuDF / Polars GPU.
