#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backends.hpp"

namespace {

// Tupla de ancho fijo en la pila del consumidor (un arreglo de tamaño variable obligaría a allocar por tupla).
// Todos los valores se ensanchan a int64_t para que el operador no dependa del tipo de la columna.
// MAX_ANCHO cubre la tupla más ancha de los planes: la salida del segundo join de Q3 (10 columnas).
constexpr int MAX_ANCHO = 12;
struct Tupla {
  int64_t v[MAX_ANCHO];
};

// next() virtual por tupla a propósito: el despacho interpretado es lo que este baseline mide frente a omp y cuda.
struct Operador {
  int ancho = 0;
  virtual ~Operador() = default;
  virtual void open() {}
  virtual bool next(Tupla& t) = 0;
};
using Op = std::unique_ptr<Operador>;

// Lee las columnas en el orden dado: primero las de int32_t, luego las de int8_t.
struct Scan : Operador {
  std::vector<const int32_t*> c32;
  std::vector<const int8_t*> c8;
  size_t n, i = 0;
  Scan(size_t n_, std::vector<const int32_t*> a, std::vector<const int8_t*> b = {})
      : c32(std::move(a)), c8(std::move(b)), n(n_) {
    ancho = static_cast<int>(c32.size() + c8.size());
  }
  void open() override { i = 0; }
  bool next(Tupla& t) override {
    if (i == n) return false;
    int k = 0;
    for (const int32_t* c : c32) t.v[k++] = c[i];
    for (const int8_t* c : c8) t.v[k++] = c[i];
    i++;
    return true;
  }
};

// Predicados y expresiones van por std::function (otra llamada indirecta por tupla, como el intérprete de expresiones de un Volcano real).
using Predicado = std::function<bool(const Tupla&)>;

struct Filter : Operador {
  Op hijo;
  Predicado pred;
  Filter(Op h, Predicado p) : hijo(std::move(h)), pred(std::move(p)) { ancho = hijo->ancho; }
  void open() override { hijo->open(); }
  bool next(Tupla& t) override {
    while (hijo->next(t))
      if (pred(t)) return true;
    return false;
  }
};

// Emite las columnas de probe seguidas de las de build.
// La tabla de build es un mapa único (no multimap): en Q3 el lado build siempre es la clave primaria (c_custkey, o_orderkey).
// std::unordered_map y no direccionamiento abierto (es el baseline; la tabla hash afinada es trabajo de omp y cuda).
struct HashJoin : Operador {
  Op build, probe;
  int col_build, col_probe;
  std::unordered_map<int64_t, Tupla> tabla;
  HashJoin(Op b, int cb, Op p, int cp) : build(std::move(b)), probe(std::move(p)), col_build(cb), col_probe(cp) {
    ancho = probe->ancho + build->ancho;
  }
  void open() override {
    tabla.clear();
    build->open();
    Tupla t;
    while (build->next(t)) tabla.emplace(t.v[col_build], t);
    probe->open();
  }
  bool next(Tupla& t) override {
    while (probe->next(t)) {
      auto it = tabla.find(t.v[col_probe]);
      if (it == tabla.end()) continue;
      for (int k = 0; k < build->ancho; k++) t.v[probe->ancho + k] = it->second.v[k];
      return true;
    }
    return false;
  }
};

// Agrupa por clave(t) y acumula con actualizar(acumulador, t); el acumulador nace en ceros.
// Bloqueante: consume todo el hijo en open() y luego emite un acumulador por grupo.
struct HashAgg : Operador {
  Op hijo;
  std::function<int64_t(const Tupla&)> clave;
  std::function<void(Tupla&, const Tupla&)> actualizar;
  std::unordered_map<int64_t, Tupla> grupos;
  std::unordered_map<int64_t, Tupla>::const_iterator it;
  HashAgg(Op h, int ancho_salida, std::function<int64_t(const Tupla&)> k, std::function<void(Tupla&, const Tupla&)> a)
      : hijo(std::move(h)), clave(std::move(k)), actualizar(std::move(a)) {
    ancho = ancho_salida;
  }
  void open() override {
    grupos.clear();
    hijo->open();
    Tupla t;
    while (hijo->next(t)) actualizar(grupos.try_emplace(clave(t), Tupla{}).first->second, t);
    it = grupos.begin();
  }
  bool next(Tupla& t) override {
    if (it == grupos.end()) return false;
    t = it->second;
    ++it;
    return true;
  }
};

}  // namespace

ResQ1 q1_serial(const Datos& d) {
  const Lineitem& l = d.l;
  // 0 quantity, 1 extendedprice, 2 discount, 3 tax, 4 shipdate, 5 returnflag, 6 linestatus
  Op plan = std::make_unique<HashAgg>(
      std::make_unique<Filter>(
          std::make_unique<Scan>(l.n,
                                 std::vector<const int32_t*>{l.quantity.data(), l.extendedprice.data(),
                                                             l.discount.data(), l.tax.data(), l.shipdate.data()},
                                 std::vector<const int8_t*>{l.returnflag.data(), l.linestatus.data()}),
          [](const Tupla& t) { return t.v[4] <= Q1_FECHA_MAX; }),
      8, [](const Tupla& t) { return t.v[5] << 8 | t.v[6]; },
      [](Tupla& a, const Tupla& t) {
        const int64_t disc_price = t.v[1] * (100 - t.v[2]);
        a.v[0] = t.v[5];
        a.v[1] = t.v[6];
        a.v[2] += t.v[0];
        a.v[3] += t.v[1];
        a.v[4] += disc_price;
        a.v[5] += disc_price * (100 + t.v[3]);
        a.v[6] += t.v[2];
        a.v[7]++;
      });
  ResQ1 r;
  plan->open();
  Tupla t;
  while (plan->next(t))
    r.push_back({static_cast<int8_t>(t.v[0]), static_cast<int8_t>(t.v[1]), t.v[2], t.v[3], t.v[4], t.v[5], t.v[6],
                 t.v[7]});
  ordenar_q1(r);
  return r;
}

ResQ3 q3_serial(const Datos& d) {
  const Customer& c = d.c;
  const Orders& o = d.o;
  const Lineitem& l = d.l;
  // customer: 0 custkey, 1 mktsegment
  Op clientes = std::make_unique<Filter>(
      std::make_unique<Scan>(c.n, std::vector<const int32_t*>{c.custkey.data()},
                             std::vector<const int8_t*>{c.mktsegment.data()}),
      [](const Tupla& t) { return t.v[1] == SEGMENTO_BUILDING; });
  // orders: 0 orderkey, 1 custkey, 2 orderdate, 3 shippriority; tras el join, 4-5 de customer
  Op pedidos = std::make_unique<Filter>(
      std::make_unique<Scan>(o.n, std::vector<const int32_t*>{o.orderkey.data(), o.custkey.data(),
                                                              o.orderdate.data(), o.shippriority.data()}),
      [](const Tupla& t) { return t.v[2] < Q3_FECHA; });
  Op pedidos_building = std::make_unique<HashJoin>(std::move(clientes), 0, std::move(pedidos), 1);
  // lineitem: 0 orderkey, 1 extendedprice, 2 discount, 3 shipdate; tras el join, 4-9 de pedidos_building
  Op lineas = std::make_unique<Filter>(
      std::make_unique<Scan>(l.n, std::vector<const int32_t*>{l.orderkey.data(), l.extendedprice.data(),
                                                              l.discount.data(), l.shipdate.data()}),
      [](const Tupla& t) { return t.v[3] > Q3_FECHA; });
  // o_orderdate y o_shippriority dependen funcionalmente de l_orderkey, así que basta agrupar por la clave.
  Op plan = std::make_unique<HashAgg>(
      std::make_unique<HashJoin>(std::move(pedidos_building), 0, std::move(lineas), 0), 4,
      [](const Tupla& t) { return t.v[0]; },
      [](Tupla& a, const Tupla& t) {
        a.v[0] = t.v[0];
        a.v[1] += t.v[1] * (100 - t.v[2]);
        a.v[2] = t.v[6];
        a.v[3] = t.v[7];
      });
  ResQ3 r;
  plan->open();
  Tupla t;
  while (plan->next(t))
    r.push_back({static_cast<int32_t>(t.v[0]), t.v[1], static_cast<int32_t>(t.v[2]), static_cast<int32_t>(t.v[3])});
  top_q3(r);
  return r;
}

ResQ6 q6_serial(const Datos& d) {
  const Lineitem& l = d.l;
  // 0 shipdate, 1 discount, 2 quantity, 3 extendedprice
  Op plan = std::make_unique<HashAgg>(
      std::make_unique<Filter>(
          std::make_unique<Scan>(l.n, std::vector<const int32_t*>{l.shipdate.data(), l.discount.data(),
                                                                  l.quantity.data(), l.extendedprice.data()}),
          [](const Tupla& t) {
            return t.v[0] >= Q6_FECHA_MIN && t.v[0] < Q6_FECHA_MAX && t.v[1] >= Q6_DESC_MIN &&
                   t.v[1] <= Q6_DESC_MAX && t.v[2] < Q6_CANT_MAX;
          }),
      1, [](const Tupla&) { return int64_t{0}; },
      [](Tupla& a, const Tupla& t) { a.v[0] += t.v[3] * t.v[1]; });
  plan->open();
  Tupla t;
  return plan->next(t) ? t.v[0] : 0;
}
