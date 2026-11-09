#include "datos.hpp"

#include <cstdio>
#include <cstdlib>

template <typename T>
static std::vector<T> leer(const std::string& ruta, size_t& n) {
  FILE* f = std::fopen(ruta.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "No se pudo abrir %s (¿corriste scripts/gen_datos.py?)\n", ruta.c_str());
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  n = std::ftell(f) / sizeof(T);
  std::fseek(f, 0, SEEK_SET);
  std::vector<T> v(n);
  if (std::fread(v.data(), sizeof(T), n, f) != n) {
    std::fprintf(stderr, "Lectura incompleta de %s\n", ruta.c_str());
    std::exit(1);
  }
  std::fclose(f);
  return v;
}

Datos cargar_datos(const std::string& dir) {
  Datos d;
  auto col = [&](const char* tabla, const char* nombre) { return dir + "/" + tabla + "/" + nombre + ".col"; };
  size_t& nl = d.l.n;
  d.l.orderkey = leer<int32_t>(col("lineitem", "l_orderkey"), nl);
  d.l.quantity = leer<int32_t>(col("lineitem", "l_quantity"), nl);
  d.l.extendedprice = leer<int32_t>(col("lineitem", "l_extendedprice"), nl);
  d.l.discount = leer<int32_t>(col("lineitem", "l_discount"), nl);
  d.l.tax = leer<int32_t>(col("lineitem", "l_tax"), nl);
  d.l.shipdate = leer<int32_t>(col("lineitem", "l_shipdate"), nl);
  d.l.returnflag = leer<int8_t>(col("lineitem", "l_returnflag"), nl);
  d.l.linestatus = leer<int8_t>(col("lineitem", "l_linestatus"), nl);
  size_t& no = d.o.n;
  d.o.orderkey = leer<int32_t>(col("orders", "o_orderkey"), no);
  d.o.custkey = leer<int32_t>(col("orders", "o_custkey"), no);
  d.o.orderdate = leer<int32_t>(col("orders", "o_orderdate"), no);
  d.o.shippriority = leer<int32_t>(col("orders", "o_shippriority"), no);
  size_t& nc = d.c.n;
  d.c.custkey = leer<int32_t>(col("customer", "c_custkey"), nc);
  d.c.mktsegment = leer<int8_t>(col("customer", "c_mktsegment"), nc);
  return d;
}
