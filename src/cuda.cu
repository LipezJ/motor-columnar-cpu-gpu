// Backend GPU por tiles (Shanbhag, Madden y Yu, "A Study of the Fundamental Performance Characteristics of GPUs
// and CPUs for Database Analytics", SIGMOD 2020; el código de referencia se llama Crystal).
//
// Medición: transferencia_ms y computo_ms salen de cudaEvent y excluyen la creación del contexto CUDA
// (ocurre en el primer cudaEventCreate, antes de abrir cualquier intervalo).
// total_ms de la repetición 0 sí incluye esa creación, y su computo_ms incluye la carga perezosa del módulo
// en el primer lanzamiento de cada kernel (CUDA_MODULE_LOADING=LAZY por defecto desde CUDA 12).
// Cada llamada copia sus columnas y libera todo al salir: transferencia_ms + computo_ms es el caso de datos no
// residentes en la GPU y computo_ms solo es el caso de datos residentes.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <cuda_runtime.h>

#include "backends.hpp"

static void chequear(cudaError_t e, const char* expr, const char* archivo, int linea) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "Error CUDA en %s:%d: %s: %s\n", archivo, linea, expr, cudaGetErrorString(e));
    std::exit(4);
  }
}
#define CK(x) chequear((x), #x, __FILE__, __LINE__)
// Los errores de lanzamiento salen aquí; los de ejecución, en el siguiente cudaEventSynchronize o cudaMemcpy.
#define CK_KERNEL() CK(cudaGetLastError())

// Valores de Crystal para SSB.
// reducir_bloque asume BLOCK_THREADS múltiplo de 32 y como mucho 32 warps.
constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;

static unsigned bloques(int64_t n) {
  return static_cast<unsigned>(std::max<int64_t>(1, (n + TILE - 1) / TILE));
}

// Fila k del hilo t del tile: k * BLOCK_THREADS + t (reparto rayado de Crystal, para que cada warp lea 128 B
// consecutivos por columna).
__device__ __forceinline__ int fila_tile(int k) { return k * BLOCK_THREADS + threadIdx.x; }

__device__ __forceinline__ int filas_en_tile(int64_t n, int64_t base) {
  return n - base < TILE ? static_cast<int>(n - base) : TILE;
}

__device__ __forceinline__ void iniciar_sel(int n_tile, bool (&sel)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = fila_tile(k) < n_tile;
}

// Solo lee las filas que siguen seleccionadas.
// Con predicados selectivos (Q6 deja ~2 %) la mayoría de los sectores de 32 B de las columnas posteriores no se piden.
template <typename T>
__device__ __forceinline__ void cargar(const T* __restrict__ col_tile, const bool (&sel)[ITEMS_PER_THREAD],
                                       T (&r)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++)
    if (sel[k]) r[k] = col_tile[fila_tile(k)];
}

// Suma de todo el bloque, válida solo en el hilo 0.
// Todos los hilos del bloque tienen que llamarla (usa __syncthreads).
__device__ unsigned long long reducir_bloque(unsigned long long v) {
  __shared__ unsigned long long parcial[BLOCK_THREADS / 32];
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) parcial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < BLOCK_THREADS / 32 ? parcial[lane] : 0;
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
  }
  return v;
}

struct Cronometro {
  cudaEvent_t ini, fin;
  Cronometro() {
    CK(cudaEventCreate(&ini));
    CK(cudaEventCreate(&fin));
  }
  ~Cronometro() {
    cudaEventDestroy(ini);
    cudaEventDestroy(fin);
  }
  void empezar() { CK(cudaEventRecord(ini)); }
  void parar(double& acumulado) {
    CK(cudaEventRecord(fin));
    CK(cudaEventSynchronize(fin));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, ini, fin));
    acumulado += ms;
  }
};

// Copia desde memoria paginable, como quedan los std::vector de cargar_datos.
// Con memoria fijada (cudaHostRegister) la copia sería más rápida, pero se mediría un caso que el motor no tiene.
template <typename T>
static T* subir(const std::vector<T>& v) {
  T* p = nullptr;
  CK(cudaMalloc(&p, v.size() * sizeof(T)));
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

template <typename T>
static T* reservar(size_t n) {
  T* p = nullptr;
  CK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

__global__ void k_q6(const int32_t* __restrict__ fecha, const int32_t* __restrict__ desc,
                     const int32_t* __restrict__ cant, const int32_t* __restrict__ precio, int64_t n,
                     unsigned long long* res) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * TILE;
  const int n_tile = filas_en_tile(n, base);
  bool sel[ITEMS_PER_THREAD];
  int32_t f[ITEMS_PER_THREAD], d[ITEMS_PER_THREAD], q[ITEMS_PER_THREAD], p[ITEMS_PER_THREAD];
  iniciar_sel(n_tile, sel);
  cargar(fecha + base, sel, f);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && f[k] >= Q6_FECHA_MIN && f[k] < Q6_FECHA_MAX;
  cargar(desc + base, sel, d);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && d[k] >= Q6_DESC_MIN && d[k] <= Q6_DESC_MAX;
  cargar(cant + base, sel, q);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && q[k] < Q6_CANT_MAX;
  cargar(precio + base, sel, p);
  unsigned long long suma = 0;
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++)
    if (sel[k]) suma += static_cast<unsigned long long>(p[k]) * static_cast<unsigned long long>(d[k]);
  suma = reducir_bloque(suma);
  if (threadIdx.x == 0 && suma) atomicAdd(res, suma);
}

ResQ6 q6_cuda(const Datos& d, Tiempos& t) {
  const Lineitem& l = d.l;
  Cronometro c;
  c.empezar();
  int32_t* fecha = subir(l.shipdate);
  int32_t* desc = subir(l.discount);
  int32_t* cant = subir(l.quantity);
  int32_t* precio = subir(l.extendedprice);
  unsigned long long* res = reservar<unsigned long long>(1);
  c.parar(t.transferencia_ms);

  c.empezar();
  CK(cudaMemset(res, 0, sizeof(unsigned long long)));
  k_q6<<<bloques(l.n), BLOCK_THREADS>>>(fecha, desc, cant, precio, static_cast<int64_t>(l.n), res);
  CK_KERNEL();
  c.parar(t.computo_ms);

  unsigned long long h = 0;
  c.empezar();
  CK(cudaMemcpy(&h, res, sizeof h, cudaMemcpyDeviceToHost));
  c.parar(t.transferencia_ms);

  for (void* p : {static_cast<void*>(fecha), static_cast<void*>(desc), static_cast<void*>(cant),
                  static_cast<void*>(precio), static_cast<void*>(res)})
    CK(cudaFree(p));
  return static_cast<ResQ6>(h);
}

// Ranura de (returnflag, linestatus): TPC-H solo genera returnflag en {A, N, R} y linestatus en {F, O}.
// Cualquier otro valor caería en la ranura de R o de O.
constexpr int GRUPOS_Q1 = 6;
constexpr int AGREGADOS_Q1 = 6;  // sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count
__device__ __forceinline__ int grupo_q1(int8_t rf, int8_t ls) {
  return (rf == 'A' ? 0 : rf == 'N' ? 1 : 2) * 2 + (ls == 'O');
}

// Todas las filas caen en 4 grupos.
// Con INGENUO, cada fila hace 6 atomicAdd sobre las mismas 24 direcciones globales, y todos los SM de la GPU se
// serializan en esas líneas de L2.
// Sin INGENUO, las filas acumulan con atómicos en memoria compartida, que solo compiten dentro del bloque y se
// resuelven en el SM.
// Después el bloque hace como mucho 36 atómicos globales por cada TILE filas.
// ponytail: un atómico compartido por fila; si el perfil muestra contención en compartida, agregar antes por warp
// con __match_any_sync.
template <bool INGENUO>
__global__ void k_q1(const int32_t* __restrict__ cant, const int32_t* __restrict__ precio,
                     const int32_t* __restrict__ desc, const int32_t* __restrict__ imp,
                     const int32_t* __restrict__ fecha, const int8_t* __restrict__ rf,
                     const int8_t* __restrict__ ls, int64_t n, unsigned long long* glob) {
  __shared__ unsigned long long agg[GRUPOS_Q1 * AGREGADOS_Q1];
  if constexpr (!INGENUO) {
    for (int j = threadIdx.x; j < GRUPOS_Q1 * AGREGADOS_Q1; j += BLOCK_THREADS) agg[j] = 0;
    __syncthreads();
  }
  unsigned long long* destino = INGENUO ? glob : agg;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * TILE;
  const int n_tile = filas_en_tile(n, base);
  bool sel[ITEMS_PER_THREAD];
  int32_t f[ITEMS_PER_THREAD], q[ITEMS_PER_THREAD], p[ITEMS_PER_THREAD], d[ITEMS_PER_THREAD], x[ITEMS_PER_THREAD];
  int8_t r[ITEMS_PER_THREAD], s[ITEMS_PER_THREAD];
  iniciar_sel(n_tile, sel);
  cargar(fecha + base, sel, f);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && f[k] <= Q1_FECHA_MAX;
  cargar(cant + base, sel, q);
  cargar(precio + base, sel, p);
  cargar(desc + base, sel, d);
  cargar(imp + base, sel, x);
  cargar(rf + base, sel, r);
  cargar(ls + base, sel, s);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) {
    if (!sel[k]) continue;
    unsigned long long* a = destino + grupo_q1(r[k], s[k]) * AGREGADOS_Q1;
    const unsigned long long precio_desc = static_cast<unsigned long long>(p[k]) * (100 - d[k]);
    atomicAdd(a + 0, static_cast<unsigned long long>(q[k]));
    atomicAdd(a + 1, static_cast<unsigned long long>(p[k]));
    atomicAdd(a + 2, precio_desc);
    atomicAdd(a + 3, precio_desc * (100 + x[k]));
    atomicAdd(a + 4, static_cast<unsigned long long>(d[k]));
    atomicAdd(a + 5, 1ull);
  }

  if constexpr (!INGENUO) {
    __syncthreads();
    for (int j = threadIdx.x; j < GRUPOS_Q1 * AGREGADOS_Q1; j += BLOCK_THREADS)
      if (agg[j]) atomicAdd(glob + j, agg[j]);
  }
}

ResQ1 q1_cuda(const Datos& d, Tiempos& t) {
  // MOTOR_Q1_INGENUO=1 cambia a un atómico global por fila, para medir la contención que evita la agregación en
  // memoria compartida. Cualquier otro valor, o no definirla, usa la versión por bloque.
  const char* env = std::getenv("MOTOR_Q1_INGENUO");
  const bool ingenuo = env && std::strcmp(env, "1") == 0;
  const Lineitem& l = d.l;
  Cronometro c;
  c.empezar();
  int32_t* cant = subir(l.quantity);
  int32_t* precio = subir(l.extendedprice);
  int32_t* desc = subir(l.discount);
  int32_t* imp = subir(l.tax);
  int32_t* fecha = subir(l.shipdate);
  int8_t* rf = subir(l.returnflag);
  int8_t* ls = subir(l.linestatus);
  unsigned long long* glob = reservar<unsigned long long>(GRUPOS_Q1 * AGREGADOS_Q1);
  c.parar(t.transferencia_ms);

  c.empezar();
  CK(cudaMemset(glob, 0, GRUPOS_Q1 * AGREGADOS_Q1 * sizeof(unsigned long long)));
  const int64_t n = static_cast<int64_t>(l.n);
  if (ingenuo) k_q1<true><<<bloques(n), BLOCK_THREADS>>>(cant, precio, desc, imp, fecha, rf, ls, n, glob);
  else k_q1<false><<<bloques(n), BLOCK_THREADS>>>(cant, precio, desc, imp, fecha, rf, ls, n, glob);
  CK_KERNEL();
  c.parar(t.computo_ms);

  unsigned long long h[GRUPOS_Q1 * AGREGADOS_Q1];
  c.empezar();
  CK(cudaMemcpy(h, glob, sizeof h, cudaMemcpyDeviceToHost));
  c.parar(t.transferencia_ms);

  for (void* p : {static_cast<void*>(cant), static_cast<void*>(precio), static_cast<void*>(desc),
                  static_cast<void*>(imp), static_cast<void*>(fecha), static_cast<void*>(rf),
                  static_cast<void*>(ls), static_cast<void*>(glob)})
    CK(cudaFree(p));

  ResQ1 r;
  for (int g = 0; g < GRUPOS_Q1; g++) {
    const unsigned long long* a = h + g * AGREGADOS_Q1;
    if (a[5] == 0) continue;
    FilaQ1 f;
    f.returnflag = "ANR"[g / 2];
    f.linestatus = "FO"[g % 2];
    f.sum_qty = static_cast<int64_t>(a[0]);
    f.sum_base_price = static_cast<int64_t>(a[1]);
    f.sum_disc_price = static_cast<int64_t>(a[2]);
    f.sum_charge = static_cast<int64_t>(a[3]);
    f.sum_disc = static_cast<int64_t>(a[4]);
    f.count = static_cast<int64_t>(a[5]);
    r.push_back(f);
  }
  ordenar_q1(r);
  return r;
}

// Tabla hash del lado build: orders filtrados, indexada por o_orderkey.
// Direccionamiento abierto con sondeo lineal en tres arreglos paralelos de capacidad potencia de 2:
//   claves  o_orderkey, o VACIO si la ranura está libre;
//   filas   posición de la fila en orders, para leer o_orderdate y o_shippriority solo de los grupos finales;
//   revenue suma de l_extendedprice * (100 - l_discount) de las líneas que encajan (escala 1e4).
// Direccionamiento abierto y no encadenamiento: se construye sin reservar memoria en el kernel, insertando con un
// atomicCAS sobre la clave, como el hash join de Crystal.
// Sondeo lineal: la ranura siguiente suele estar en el mismo sector de 32 B, así que una colisión casi nunca cuesta
// otra transacción de memoria.
// revenue en la misma tabla y no en una tabla de grupos aparte (groupjoin, Moerkotte y Neumann, VLDB 2011):
// o_orderkey es clave primaria de orders, así que cada ranura de build es exactamente un grupo de GROUP BY l_orderkey.
constexpr int32_t VACIO = -1;  // las claves de TPC-H empiezan en 1; cudaMemset(0xFF) deja VACIO en cada ranura

// Hash de Fibonacci (Knuth, TAOCP vol. 3, sección 6.4): toma los bits altos del producto.
// Los bits altos dependen de todos los bits de la clave; o_orderkey es denso en grupos de 8 de cada 32.
__device__ __forceinline__ uint32_t hash_clave(int32_t k, int bits) {
  return static_cast<uint32_t>((static_cast<uint64_t>(static_cast<uint32_t>(k)) * 0x9E3779B97F4A7C15ull) >>
                               (64 - bits));
}

__global__ void k_q3_clientes(const int32_t* __restrict__ custkey, const int8_t* __restrict__ seg, int64_t n,
                              uint32_t* bitmap) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * TILE;
  const int n_tile = filas_en_tile(n, base);
  bool sel[ITEMS_PER_THREAD];
  int8_t s[ITEMS_PER_THREAD];
  int32_t c[ITEMS_PER_THREAD];
  iniciar_sel(n_tile, sel);
  cargar(seg + base, sel, s);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && s[k] == SEGMENTO_BUILDING;
  cargar(custkey + base, sel, c);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++)
    if (sel[k]) atomicOr(bitmap + (c[k] >> 5), 1u << (c[k] & 31));
}

__global__ void k_q3_build(const int32_t* __restrict__ orderkey, const int32_t* __restrict__ custkey,
                           const int32_t* __restrict__ orderdate, int64_t n, const uint32_t* __restrict__ bitmap,
                           int32_t max_custkey, int32_t* claves, int32_t* filas, uint32_t mascara, int bits) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * TILE;
  const int n_tile = filas_en_tile(n, base);
  bool sel[ITEMS_PER_THREAD];
  int32_t f[ITEMS_PER_THREAD], c[ITEMS_PER_THREAD], o[ITEMS_PER_THREAD];
  iniciar_sel(n_tile, sel);
  cargar(orderdate + base, sel, f);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && f[k] < Q3_FECHA;
  cargar(custkey + base, sel, c);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++)
    sel[k] = sel[k] && c[k] >= 0 && c[k] <= max_custkey && (bitmap[c[k] >> 5] >> (c[k] & 31) & 1u);
  cargar(orderkey + base, sel, o);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) {
    if (!sel[k]) continue;
    // Termina porque la capacidad supera el número de filas de orders, así que siempre queda una ranura VACIO.
    for (uint32_t r = hash_clave(o[k], bits);; r = (r + 1) & mascara) {
      if (atomicCAS(claves + r, VACIO, o[k]) == VACIO) {
        filas[r] = static_cast<int32_t>(base + fila_tile(k));
        break;
      }
    }
  }
}

// Un atomicAdd global por línea es aceptable aquí, a diferencia de Q1: los grupos son decenas de miles por unidad
// de SF y cada uno recibe como mucho 7 líneas, así que casi no hay dos hilos sobre la misma dirección.
__global__ void k_q3_probe(const int32_t* __restrict__ orderkey, const int32_t* __restrict__ precio,
                           const int32_t* __restrict__ desc, const int32_t* __restrict__ shipdate, int64_t n,
                           const int32_t* __restrict__ claves, unsigned long long* revenue, uint32_t mascara,
                           int bits) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * TILE;
  const int n_tile = filas_en_tile(n, base);
  bool sel[ITEMS_PER_THREAD];
  int32_t f[ITEMS_PER_THREAD], o[ITEMS_PER_THREAD], p[ITEMS_PER_THREAD], d[ITEMS_PER_THREAD];
  int32_t ranura[ITEMS_PER_THREAD];
  iniciar_sel(n_tile, sel);
  cargar(shipdate + base, sel, f);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) sel[k] = sel[k] && f[k] > Q3_FECHA;
  cargar(orderkey + base, sel, o);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++) {
    ranura[k] = -1;
    if (!sel[k]) continue;
    for (uint32_t r = hash_clave(o[k], bits);; r = (r + 1) & mascara) {
      const int32_t c = claves[r];
      if (c == o[k]) {
        ranura[k] = static_cast<int32_t>(r);
        break;
      }
      if (c == VACIO) break;
    }
    sel[k] = ranura[k] >= 0;
  }
  cargar(precio + base, sel, p);
  cargar(desc + base, sel, d);
#pragma unroll
  for (int k = 0; k < ITEMS_PER_THREAD; k++)
    if (sel[k]) atomicAdd(revenue + ranura[k], static_cast<unsigned long long>(p[k]) * (100 - d[k]));
}

// Un grupo existe si y solo si su revenue es positivo: en TPC-H l_extendedprice > 0 y l_discount <= 10, así que
// toda línea que encaja suma algo.
// ponytail: un atomicAdd por grupo sobre n_grupos; agregar por warp (__ballot_sync + __popc) si pesa en el perfil.
__global__ void k_q3_recolectar(const int32_t* __restrict__ claves, const int32_t* __restrict__ filas,
                                const unsigned long long* __restrict__ revenue, int64_t capacidad,
                                const int32_t* __restrict__ orderdate, const int32_t* __restrict__ shippriority,
                                FilaQ3* grupos, unsigned* n_grupos) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;
  if (r >= capacidad || revenue[r] == 0) return;
  const int32_t fila = filas[r];
  FilaQ3 g;
  g.orderkey = claves[r];
  g.revenue = static_cast<int64_t>(revenue[r]);
  g.orderdate = orderdate[fila];
  g.shippriority = shippriority[fila];
  grupos[atomicAdd(n_grupos, 1u)] = g;
}

ResQ3 q3_cuda(const Datos& d, Tiempos& t) {
  const Lineitem& l = d.l;
  const Orders& o = d.o;
  const Customer& cl = d.c;

  // El tamaño del bitmap sale del máximo c_custkey, que un catálogo real tendría en sus estadísticas de columna.
  // Se calcula en el host y fuera de lo medido.
  const int32_t max_custkey = cl.n ? *std::max_element(cl.custkey.begin(), cl.custkey.end()) : 0;
  const size_t palabras_bitmap = static_cast<size_t>(std::max<int32_t>(max_custkey, 0)) / 32 + 1;

  // Capacidad: menor potencia de 2 mayor que o.n.
  // Con la selectividad de Q3 (~10 % de orders) la carga queda entre 5 % y 10 %.
  // Mayor que o.n y no que las filas filtradas: así no hace falta contar antes de construir, y el sondeo termina
  // aunque el filtro dejara pasar todo orders.
  int bits = 1;
  while ((static_cast<uint64_t>(1) << bits) <= o.n) bits++;
  const uint64_t capacidad = static_cast<uint64_t>(1) << bits;
  const uint32_t mascara = static_cast<uint32_t>(capacidad - 1);

  Cronometro c;
  c.empezar();
  int32_t* c_custkey = subir(cl.custkey);
  int8_t* c_seg = subir(cl.mktsegment);
  int32_t* o_orderkey = subir(o.orderkey);
  int32_t* o_custkey = subir(o.custkey);
  int32_t* o_orderdate = subir(o.orderdate);
  int32_t* o_shippriority = subir(o.shippriority);
  int32_t* l_orderkey = subir(l.orderkey);
  int32_t* l_precio = subir(l.extendedprice);
  int32_t* l_desc = subir(l.discount);
  int32_t* l_shipdate = subir(l.shipdate);
  uint32_t* bitmap = reservar<uint32_t>(palabras_bitmap);
  int32_t* claves = reservar<int32_t>(capacidad);
  int32_t* filas = reservar<int32_t>(capacidad);
  unsigned long long* revenue = reservar<unsigned long long>(capacidad);
  // Cota superior de grupos: cada grupo es una fila distinta de orders.
  FilaQ3* grupos = reservar<FilaQ3>(std::max<size_t>(o.n, 1));
  unsigned* n_grupos = reservar<unsigned>(1);
  c.parar(t.transferencia_ms);

  c.empezar();
  CK(cudaMemset(bitmap, 0, palabras_bitmap * sizeof(uint32_t)));
  CK(cudaMemset(claves, 0xFF, capacidad * sizeof(int32_t)));
  CK(cudaMemset(revenue, 0, capacidad * sizeof(unsigned long long)));
  CK(cudaMemset(n_grupos, 0, sizeof(unsigned)));
  k_q3_clientes<<<bloques(cl.n), BLOCK_THREADS>>>(c_custkey, c_seg, static_cast<int64_t>(cl.n), bitmap);
  CK_KERNEL();
  k_q3_build<<<bloques(o.n), BLOCK_THREADS>>>(o_orderkey, o_custkey, o_orderdate, static_cast<int64_t>(o.n), bitmap,
                                              max_custkey, claves, filas, mascara, bits);
  CK_KERNEL();
  k_q3_probe<<<bloques(l.n), BLOCK_THREADS>>>(l_orderkey, l_precio, l_desc, l_shipdate, static_cast<int64_t>(l.n),
                                              claves, revenue, mascara, bits);
  CK_KERNEL();
  const unsigned bloques_recolectar = static_cast<unsigned>((capacidad + BLOCK_THREADS - 1) / BLOCK_THREADS);
  k_q3_recolectar<<<bloques_recolectar, BLOCK_THREADS>>>(claves, filas, revenue, static_cast<int64_t>(capacidad),
                                                         o_orderdate, o_shippriority, grupos, n_grupos);
  CK_KERNEL();
  c.parar(t.computo_ms);

  unsigned h_n = 0;
  ResQ3 r;
  c.empezar();
  CK(cudaMemcpy(&h_n, n_grupos, sizeof h_n, cudaMemcpyDeviceToHost));
  r.resize(h_n);
  if (h_n) CK(cudaMemcpy(r.data(), grupos, h_n * sizeof(FilaQ3), cudaMemcpyDeviceToHost));
  c.parar(t.transferencia_ms);

  for (void* p : {static_cast<void*>(c_custkey), static_cast<void*>(c_seg), static_cast<void*>(o_orderkey),
                  static_cast<void*>(o_custkey), static_cast<void*>(o_orderdate), static_cast<void*>(o_shippriority),
                  static_cast<void*>(l_orderkey), static_cast<void*>(l_precio), static_cast<void*>(l_desc),
                  static_cast<void*>(l_shipdate), static_cast<void*>(bitmap), static_cast<void*>(claves),
                  static_cast<void*>(filas), static_cast<void*>(revenue), static_cast<void*>(grupos),
                  static_cast<void*>(n_grupos)})
    CK(cudaFree(p));

  top_q3(r);
  return r;
}
