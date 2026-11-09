#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "backends.hpp"

// Paralelismo por morsels (Leis et al., SIGMOD 2014): cada hilo toma rangos de TAM_MORSEL filas con
// schedule(dynamic) y acumula en estado propio; los estados se combinan una sola vez al final.
// TAM_MORSEL son filas de lineitem u orders.
// Leis et al. usan ~100 000 tuplas: basta para amortizar el reparto dinámico y deja ~90 morsels en SF1 para equilibrar 15 hilos.
constexpr size_t TAM_MORSEL = 1 << 16;

static size_t num_morsels(size_t n) { return (n + TAM_MORSEL - 1) / TAM_MORSEL; }

// Arreglo de n copias de v inicializado por todos los hilos con schedule(static).
// Linux coloca cada página en el nodo NUMA del hilo que la escribe primero (first touch).
// Con un std::vector la escribiría entera el hilo que lo construye, y en una máquina de dos sockets todos los accesos
// del otro socket serían remotos (en smexa, con la tabla en un solo nodo, Q3 empeoraba de 114 ms con 64 hilos a 135 ms con 128).
template <typename T>
static std::unique_ptr<T[]> arreglo_repartido(size_t n, T v) {
  std::unique_ptr<T[]> p(new T[n]);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) p[i] = v;
  return p;
}

ResQ6 q6_omp(const Datos& d) {
  const auto& l = d.l;
  const int32_t *fecha = l.shipdate.data(), *desc = l.discount.data(), *cant = l.quantity.data(),
                *precio = l.extendedprice.data();
  int64_t total = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total)
  for (size_t m = 0; m < num_morsels(l.n); m++) {
    const size_t ini = m * TAM_MORSEL, fin = std::min(l.n, ini + TAM_MORSEL);
    int64_t s = 0;
    // El predicado se evalúa como máscara y multiplica al producto (sin rama, el compilador vectoriza el bucle).
    // Con ramas, la selectividad de ~2 % hace que el predictor falle en cada fila que pasa.
    for (size_t i = ini; i < fin; i++) {
      const int64_t pasa = (fecha[i] >= Q6_FECHA_MIN) & (fecha[i] < Q6_FECHA_MAX) & (desc[i] >= Q6_DESC_MIN) &
                           (desc[i] <= Q6_DESC_MAX) & (cant[i] < Q6_CANT_MAX);
      s += pasa * (static_cast<int64_t>(precio[i]) * desc[i]);
    }
    total += s;
  }
  return total;
}

// Código de grupo de Q1: (returnflag % 3) * 2 + (linestatus == 'O').
// Solo es inyectivo para los alfabetos de TPC-H: returnflag en {A, N, R} y linestatus en {F, O}.
constexpr int GRUPOS_Q1 = 6;
constexpr int8_t FLAG_DE_CODIGO[3] = {'N', 'R', 'A'};  // inversa de returnflag % 3

ResQ1 q1_omp(const Datos& d) {
  const auto& l = d.l;
  const int hilos = omp_get_max_threads();
  // Un juego de agregados por hilo.
  // alignas(64) evita que dos hilos compartan una línea de caché (false sharing).
  struct alignas(64) Local {
    FilaQ1 g[GRUPOS_Q1] = {};
  };
  std::vector<Local> locales(hilos);

#pragma omp parallel
  {
    FilaQ1* g = locales[omp_get_thread_num()].g;
#pragma omp for schedule(dynamic, 1)
    for (size_t m = 0; m < num_morsels(l.n); m++) {
      const size_t ini = m * TAM_MORSEL, fin = std::min(l.n, ini + TAM_MORSEL);
      for (size_t i = ini; i < fin; i++) {
        if (l.shipdate[i] > Q1_FECHA_MAX) continue;
        FilaQ1& f = g[(l.returnflag[i] % 3) * 2 + (l.linestatus[i] == 'O')];
        const int64_t precio = l.extendedprice[i], desc = l.discount[i];
        const int64_t con_desc = precio * (100 - desc);
        f.sum_qty += l.quantity[i];
        f.sum_base_price += precio;
        f.sum_disc_price += con_desc;
        f.sum_charge += con_desc * (100 + l.tax[i]);
        f.sum_disc += desc;
        f.count++;
      }
    }
  }

  ResQ1 r;
  for (int c = 0; c < GRUPOS_Q1; c++) {
    FilaQ1 t = {};
    t.returnflag = FLAG_DE_CODIGO[c / 2];
    t.linestatus = c % 2 ? 'O' : 'F';
    for (const auto& loc : locales) {
      const FilaQ1& f = loc.g[c];
      t.sum_qty += f.sum_qty;
      t.sum_base_price += f.sum_base_price;
      t.sum_disc_price += f.sum_disc_price;
      t.sum_charge += f.sum_charge;
      t.sum_disc += f.sum_disc;
      t.count += f.count;
    }
    if (t.count) r.push_back(t);
  }
  ordenar_q1(r);
  return r;
}

// Hash multiplicativo de Fibonacci (Knuth, TAOCP vol. 3, 6.4).
// Las o_orderkey de TPC-H son dispersas (8 de cada 32), así que el módulo directo agruparía las claves en pocos huecos.
static inline size_t hash_clave(int32_t k, int bits) {
  return static_cast<size_t>((static_cast<uint64_t>(static_cast<uint32_t>(k)) * 0x9E3779B97F4A7C15ull) >> (64 - bits));
}

ResQ3 q3_omp(const Datos& d) {
  const auto& c = d.c;
  const auto& o = d.o;
  const auto& l = d.l;

  // c_custkey es densa (1..150 000 * SF en TPC-H): un arreglo indexado por clave sustituye a un conjunto hash.
  int32_t max_cliente = 0;
#pragma omp parallel for reduction(max : max_cliente)
  for (size_t i = 0; i < c.n; i++) max_cliente = std::max(max_cliente, c.custkey[i]);
  std::vector<uint8_t> es_building(static_cast<size_t>(max_cliente) + 1, 0);
#pragma omp parallel for
  for (size_t i = 0; i < c.n; i++)
    if (c.mktsegment[i] == SEGMENTO_BUILDING) es_building[c.custkey[i]] = 1;

  auto pasa_orden = [&](size_t i) {
    const int32_t ck = o.custkey[i];
    return o.orderdate[i] < Q3_FECHA && ck >= 0 && ck <= max_cliente && es_building[ck];
  };

  // Contar primero deja la tabla al tamaño justo (factor de carga <= 0,5).
  // En SF1 pasan 147 126 de 1 500 000 orders: la tabla ocupa 8 MB en vez de los 64 MB que daría dimensionarla con o.n.
  size_t n_build = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : n_build)
  for (size_t m = 0; m < num_morsels(o.n); m++) {
    const size_t ini = m * TAM_MORSEL, fin = std::min(o.n, ini + TAM_MORSEL);
    for (size_t i = ini; i < fin; i++) n_build += pasa_orden(i);
  }

  // Tabla hash del lado build: direccionamiento abierto con sondeo lineal en arreglos paralelos (clave, fila de orders, revenue).
  // No es std::unordered_map (un nodo en el heap por entrada y un acceso indirecto más por búsqueda, y no admite inserción concurrente).
  // La clave 0 marca hueco vacío (o_orderkey empieza en 1).
  int bits = 4;
  while ((size_t{1} << bits) < 2 * n_build) bits++;
  const size_t cap = size_t{1} << bits, mascara = cap - 1;
  // filas no se inicializa: solo se lee en los huecos que el build ocupó.
  auto claves = arreglo_repartido<int32_t>(cap, 0);
  auto revenue = arreglo_repartido<int64_t>(cap, 0);
  std::unique_ptr<int32_t[]> filas(new int32_t[cap]);

  // Build paralelo con CAS sobre la clave del hueco, como el build con atomicCAS de Crystal (Shanbhag et al., SIGMOD 2020).
  // Se prefiere al particionado (radix join) porque evita un pase extra de reparto sobre orders y el build es ~1/40 de las filas de lineitem.
  // filas[h] se escribe sin atómico: solo el hilo que ganó el CAS escribe ese hueco, y la barrera implícita al final del for lo publica antes del probe.
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t m = 0; m < num_morsels(o.n); m++) {
    const size_t ini = m * TAM_MORSEL, fin = std::min(o.n, ini + TAM_MORSEL);
    for (size_t i = ini; i < fin; i++) {
      if (!pasa_orden(i)) continue;
      int32_t k = o.orderkey[i];
      for (size_t h = hash_clave(k, bits);; h = (h + 1) & mascara) {
        int32_t vacio = 0;
        if (__atomic_compare_exchange_n(&claves[h], &vacio, k, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
          filas[h] = static_cast<int32_t>(i);
          break;
        }
      }
    }
  }

  // Probe: la suma por orderkey va con fetch_add sobre revenue[h].
  // En SF1 solo 30 519 filas de lineitem suman, repartidas en 11 620 orderkeys (la contención es despreciable).
  // Una tabla de revenue por hilo costaría cap * 8 B * hilos de memoria y un merge de cap * hilos sumas.
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t m = 0; m < num_morsels(l.n); m++) {
    const size_t ini = m * TAM_MORSEL, fin = std::min(l.n, ini + TAM_MORSEL);
    for (size_t i = ini; i < fin; i++) {
      if (l.shipdate[i] <= Q3_FECHA) continue;
      const int32_t k = l.orderkey[i];
      for (size_t h = hash_clave(k, bits);; h = (h + 1) & mascara) {
        const int32_t kh = claves[h];
        if (kh == k) {
          __atomic_fetch_add(&revenue[h], static_cast<int64_t>(l.extendedprice[i]) * (100 - l.discount[i]),
                             __ATOMIC_RELAXED);
          break;
        }
        if (kh == 0) break;
      }
    }
  }

  // Cada hilo se queda con su top local; el top global está contenido en la unión de los locales.
  ResQ3 r;
#pragma omp parallel
  {
    ResQ3 local;
#pragma omp for schedule(static) nowait
    for (size_t h = 0; h < cap; h++)
      if (revenue[h] > 0) local.push_back({claves[h], revenue[h], o.orderdate[filas[h]], o.shippriority[filas[h]]});
    top_q3(local);
#pragma omp critical
    r.insert(r.end(), local.begin(), local.end());
  }
  top_q3(r);
  return r;
}
