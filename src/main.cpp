#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <omp.h>

#include "backends.hpp"

static void uso() {
  std::fprintf(stderr,
               "Uso: motor DIR_DATOS BACKEND CONSULTA [--hilos N] [--reps R]\n"
               "  BACKEND   serial | omp | cuda\n"
               "  CONSULTA  q1 | q3 | q6\n"
               "  --hilos   hilos de OpenMP (por defecto, los de OMP_NUM_THREADS o todos los núcleos)\n"
               "  --reps    repeticiones medidas (por defecto 5); la primera calienta cachés y cuenta igual\n"
               "Resultado en CSV por stdout. Tiempos por repetición en CSV por stderr.\n");
  std::exit(2);
}

int main(int argc, char** argv) {
  if (argc < 4) uso();
  const std::string dir = argv[1], backend = argv[2], consulta = argv[3];
  int reps = 5;
  for (int i = 4; i < argc; i++) {
    if (!std::strcmp(argv[i], "--hilos") && i + 1 < argc) omp_set_num_threads(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    else uso();
  }
#ifndef CON_CUDA
  if (backend == "cuda") {
    std::fprintf(stderr, "Compilado sin CUDA (usa make CUDA=1)\n");
    return 2;
  }
#endif
  if (backend != "serial" && backend != "omp" && backend != "cuda") uso();
  if (consulta != "q1" && consulta != "q3" && consulta != "q6") uso();

  const Datos d = cargar_datos(dir);
  ResQ1 r1;
  ResQ3 r3;
  ResQ6 r6 = 0;

  std::fprintf(stderr, "rep,hilos,total_ms,transferencia_ms,computo_ms\n");
  for (int rep = 0; rep < reps; rep++) {
    Tiempos t;
    const auto ini = std::chrono::steady_clock::now();
    if (backend == "serial") {
      if (consulta == "q1") r1 = q1_serial(d);
      else if (consulta == "q3") r3 = q3_serial(d);
      else r6 = q6_serial(d);
    } else if (backend == "omp") {
      if (consulta == "q1") r1 = q1_omp(d);
      else if (consulta == "q3") r3 = q3_omp(d);
      else r6 = q6_omp(d);
    }
#ifdef CON_CUDA
    else {
      if (consulta == "q1") r1 = q1_cuda(d, t);
      else if (consulta == "q3") r3 = q3_cuda(d, t);
      else r6 = q6_cuda(d, t);
    }
#endif
    const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ini).count();
    if (backend != "cuda") t.computo_ms = total;
    const int hilos = backend == "omp" ? omp_get_max_threads() : 1;
    std::fprintf(stderr, "%d,%d,%.3f,%.3f,%.3f\n", rep, hilos, total, t.transferencia_ms, t.computo_ms);
  }

  if (consulta == "q1") imprimir_q1(r1);
  else if (consulta == "q3") imprimir_q3(r3);
  else imprimir_q6(r6);
}
