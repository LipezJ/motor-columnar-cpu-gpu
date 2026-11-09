#pragma once

#include <cstdint>
#include <vector>

// Days from civil de Howard Hinnant (chrono-Compatible Low-Level Date Algorithms).
constexpr int32_t dias_desde_epoch(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<int32_t>(doe) - 719468;
}

// Parámetros de validación de TPC-H (los mismos que scripts/gen_datos.py, ESPERADO).

// Q1: l_shipdate <= 1998-12-01 - 90 días.
constexpr int32_t Q1_FECHA_MAX = dias_desde_epoch(1998, 12, 1) - 90;

// Q3
// Código de 'BUILDING' en c_mktsegment.
// Tiene que coincidir con la lista de c_mktsegment en scripts/gen_datos.py.
constexpr int8_t SEGMENTO_BUILDING = 1;
constexpr int32_t Q3_FECHA = dias_desde_epoch(1995, 3, 15);
constexpr int Q3_LIMITE = 10;

// Q6, con descuento y cantidad en centavos.
constexpr int32_t Q6_FECHA_MIN = dias_desde_epoch(1994, 1, 1);
constexpr int32_t Q6_FECHA_MAX = dias_desde_epoch(1995, 1, 1);  // exclusiva
constexpr int32_t Q6_DESC_MIN = 5;
constexpr int32_t Q6_DESC_MAX = 7;  // inclusiva
constexpr int32_t Q6_CANT_MAX = 2400;  // exclusiva

// Las sumas son enteras y exactas, con la escala indicada en cada campo.
// sum_charge es la primera que desborda int64: el grupo (N, O) suma ~1,1e17 por unidad de SF, así que no cabe por encima de SF ~80.
struct FilaQ1 {
  int8_t returnflag, linestatus;
  int64_t sum_qty;          // escala 1e2
  int64_t sum_base_price;   // escala 1e2
  int64_t sum_disc_price;   // escala 1e4: price · (100 - discount)
  int64_t sum_charge;       // escala 1e6: price · (100 - discount) · (100 + tax)
  int64_t sum_disc;         // escala 1e2, para avg_disc
  int64_t count;
};
// Ordenado por (returnflag, linestatus).
using ResQ1 = std::vector<FilaQ1>;

struct FilaQ3 {
  int32_t orderkey;
  int64_t revenue;  // escala 1e4
  int32_t orderdate, shippriority;
};
// Las Q3_LIMITE filas de mayor revenue, desempatadas por orderdate ascendente.
using ResQ3 = std::vector<FilaQ3>;

using ResQ6 = int64_t;  // revenue, escala 1e4

// Ordena y recorta como exige la consulta. Los backends lo llaman sobre sus grupos finales.
void ordenar_q1(ResQ1& r);
void top_q3(ResQ3& grupos);

void imprimir_q1(const ResQ1& r);
void imprimir_q3(const ResQ3& r);
void imprimir_q6(ResQ6 r);
