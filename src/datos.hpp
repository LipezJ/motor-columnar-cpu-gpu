#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Formato y escalas: los fija scripts/gen_datos.py (COLUMNAS).
// Decimales en centavos (escala 100), fechas en días desde 1970-01-01, flags en ASCII.

struct Lineitem {
  size_t n = 0;
  std::vector<int32_t> orderkey, quantity, extendedprice, discount, tax, shipdate;
  std::vector<int8_t> returnflag, linestatus;
};

struct Orders {
  size_t n = 0;
  std::vector<int32_t> orderkey, custkey, orderdate, shippriority;
};

struct Customer {
  size_t n = 0;
  std::vector<int32_t> custkey;
  std::vector<int8_t> mktsegment;
};

struct Datos {
  Lineitem l;
  Orders o;
  Customer c;
};

// Carga en RAM todas las columnas de dir (p. ej. data/sf1). Aborta si falta un archivo.
Datos cargar_datos(const std::string& dir);
