#include "consultas.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

// Civil from days de Howard Hinnant, inversa de dias_desde_epoch.
static void imprimir_fecha(int32_t z) {
  z += 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  std::printf("%04d-%02u-%02u", static_cast<int>(yoe) + era * 400 + (m <= 2), m, d);
}

// Imprime v / 10^decimales sin pasar por double (que perdería dígitos en las sumas grandes).
static void imprimir_escalado(int64_t v, int decimales) {
  int64_t div = 1;
  for (int i = 0; i < decimales; i++) div *= 10;
  if (v < 0) {
    std::putchar('-');
    v = -v;
  }
  std::printf("%" PRId64 ".%0*" PRId64, v / div, decimales, v % div);
}

void ordenar_q1(ResQ1& r) {
  std::sort(r.begin(), r.end(), [](const FilaQ1& a, const FilaQ1& b) {
    return a.returnflag != b.returnflag ? a.returnflag < b.returnflag : a.linestatus < b.linestatus;
  });
}

void top_q3(ResQ3& g) {
  auto mejor = [](const FilaQ3& a, const FilaQ3& b) {
    return a.revenue != b.revenue ? a.revenue > b.revenue : a.orderdate < b.orderdate;
  };
  const size_t k = std::min<size_t>(Q3_LIMITE, g.size());
  std::partial_sort(g.begin(), g.begin() + k, g.end(), mejor);
  g.resize(k);
}

void imprimir_q1(const ResQ1& r) {
  std::puts("l_returnflag,l_linestatus,sum_qty,sum_base_price,sum_disc_price,sum_charge,avg_qty,avg_price,avg_disc,count_order");
  for (const auto& f : r) {
    std::printf("%c,%c,", f.returnflag, f.linestatus);
    imprimir_escalado(f.sum_qty, 2);
    std::putchar(',');
    imprimir_escalado(f.sum_base_price, 2);
    std::putchar(',');
    imprimir_escalado(f.sum_disc_price, 4);
    std::putchar(',');
    imprimir_escalado(f.sum_charge, 6);
    const double n = static_cast<double>(f.count);
    std::printf(",%.17g,%.17g,%.17g,%" PRId64 "\n", f.sum_qty / 100.0 / n, f.sum_base_price / 100.0 / n,
                f.sum_disc / 100.0 / n, f.count);
  }
}

void imprimir_q3(const ResQ3& r) {
  std::puts("l_orderkey,revenue,o_orderdate,o_shippriority");
  for (const auto& f : r) {
    std::printf("%d,", f.orderkey);
    imprimir_escalado(f.revenue, 4);
    std::putchar(',');
    imprimir_fecha(f.orderdate);
    std::printf(",%d\n", f.shippriority);
  }
}

void imprimir_q6(ResQ6 r) {
  std::puts("revenue");
  imprimir_escalado(r, 4);
  std::putchar('\n');
}
