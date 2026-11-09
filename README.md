# Motor de consultas analíticas: serial vs CPU paralela vs GPU

Proyecto final de Programación Paralela (UIS).

**Pregunta:** ¿cuándo vale la pena ejecutar consultas analíticas en GPU frente a una CPU multinúcleo?

Para responderla implementamos desde cero, en C++ y CUDA, un motor de consultas columnar con tres versiones del mismo motor, y medimos cada una sobre consultas del benchmark TPC-H.

## Qué implementamos

| Versión | Archivo | Qué hace | Basado en |
|---|---|---|---|
| Serial | `src/serial.cpp` | Operadores Scan, Filter, HashJoin y HashAgg que procesan una tupla a la vez en un hilo. Es la línea base. | Modelo Volcano (Graefe, 1994) |
| CPU paralela | `src/omp.cpp` | Con OpenMP: reparto dinámico de bloques de filas (*morsels*) entre hilos, agregados locales por hilo y tabla hash construida en paralelo con operaciones atómicas. | Leis et al., SIGMOD 2014 (HyPer / DuckDB) |
| GPU | `src/cuda.cu` | Con CUDA: kernels que procesan *tiles* de filas, reducción en memoria compartida y tabla hash construida en la GPU con `atomicCAS`. | Shanbhag et al., SIGMOD 2020 (Crystal) |

El código común (carga de columnas, parámetros y formato de resultados) está en `src/datos.*`, `src/consultas.*` y `src/main.cpp`.

## Consultas

Cada una estresa algo distinto del paralelismo:

- **Q6**: filtro + suma. Limitada por ancho de banda de memoria.
- **Q1**: agregación en 4 grupos. Muchos hilos escriben en los mismos acumuladores (contención).
- **Q3**: join de tres tablas. Construcción concurrente de una tabla hash.

## Qué hace DuckDB aquí

DuckDB no ejecuta las consultas que medimos. Lo usamos solo como herramienta auxiliar:

1. **Generar los datos** de TPC-H (`scripts/gen_datos.py`), que exportamos a columnas binarias en `data/` (no versionado).
2. **Verificar la corrección:** su respuesta a cada consulta se guarda en `data/sfN/esperado/` y `scripts/validar.py` comprueba que nuestro motor da exactamente lo mismo.

## Qué medimos

- Tiempo de cada versión en cada consulta.
- Speedup y eficiencia de la versión OpenMP de 1 a N hilos.
- En GPU, el tiempo separado en copia de datos por PCIe y cómputo de los kernels, y la comparación entre GPUs (T4, L4, A100).

`scripts/bench.py` corre los experimentos y `scripts/graficas.py` genera las figuras. Los tiempos se toman dentro del binario (`std::chrono` y `cudaEvent`), alrededor de la consulta; Python solo orquesta.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install duckdb numpy matplotlib
.venv/bin/python scripts/gen_datos.py 10
make                      # serial + omp; make CUDA=1 CUDA_ARCH=sm_80 para añadir cuda
./build/motor data/sf10 omp q1 --hilos 8 | .venv/bin/python scripts/validar.py data/sf10/esperado/q1.csv
.venv/bin/python scripts/bench.py --datos data/sf10 --backends serial,omp --etiqueta mac
.venv/bin/python scripts/graficas.py resultados/*.csv
```

## Entornos

- **Google Colab Pro** (`colab.ipynb`): desarrollo de los kernels CUDA y comparación entre GPUs.
- **Clúster SC3 de la UIS** (`scripts/sc3.slurm`): mediciones finales de CPU y GPU. Antes de enviar el job hay que poner la partición y el número de núcleos.
