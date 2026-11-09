#pragma once

#include "consultas.hpp"
#include "datos.hpp"

// Solo el backend CUDA llena Tiempos: separa la copia por PCIe del tiempo de los kernels.
// En los backends de CPU main mide el tiempo total y lo pone en computo_ms.
struct Tiempos {
  double transferencia_ms = 0;
  double computo_ms = 0;
};

// Volcano: iteradores con next() virtual, una tupla a la vez, un hilo (Graefe, IEEE TKDE 1994).
ResQ1 q1_serial(const Datos& d);
ResQ3 q3_serial(const Datos& d);
ResQ6 q6_serial(const Datos& d);

// Paralelismo por morsels (Leis et al., SIGMOD 2014). Usa omp_get_max_threads() hilos.
ResQ1 q1_omp(const Datos& d);
ResQ3 q3_omp(const Datos& d);
ResQ6 q6_omp(const Datos& d);

#ifdef CON_CUDA
// Ejecución por tiles (Shanbhag et al., SIGMOD 2020). Copia a la GPU solo las columnas que usa la consulta.
ResQ1 q1_cuda(const Datos& d, Tiempos& t);
ResQ3 q3_cuda(const Datos& d, Tiempos& t);
ResQ6 q6_cuda(const Datos& d, Tiempos& t);
#endif
