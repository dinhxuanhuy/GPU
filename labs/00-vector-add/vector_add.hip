// Lab 00 — Vector Add (HIP)
// Mục tiêu: kernel HIP đầu tiên + đo memory bandwidth thực tế.
//
// Build:  make
// Run:    ./vector_add
//
// Ghi chú kiến trúc (Module 2):
//   - Block size 256 = bội số của Wave64 (4 wavefront / workgroup) -> không lãng phí lane.
//   - Vector add là MEMORY-BOUND: hiệu năng bị giới hạn bởi HBM bandwidth, không phải FLOPS.
//   - Grid-stride loop giúp 1 launch phủ được mảng lớn hơn số thread tối đa.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), \
                   __FILE__, __LINE__);                                      \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 24;  // 16M phần tử = 64 MB / buffer
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n, 0.0f);

  float *d_a, *d_b, *d_c;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));

  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  const int block = 256;
  const int grid = 4096;  // đủ lớn để phủ GPU, vẫn nhỏ hơn n/block

  // Warm-up (bỏ qua lần đầu để tránh nhiễu do khởi tạo module)
  vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
  HIP_CHECK(hipGetLastError());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start));
  vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

  // Kiểm tra kết quả
  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    if (h_c[i] != 3.0f) ++wrong;
  }

  // Traffic thực tế = đọc a + đọc b + ghi c = 3 * bytes
  const double seconds = ms * 1e-3;
  const double gbps = (3.0 * bytes) / seconds / 1e9;
  const double gflops = (static_cast<double>(n) / seconds) / 1e9;  // 1 phép cộng/phần tử

  std::printf("n=%d  block=%d  grid=%d\n", n, block, grid);
  std::printf("kernel time : %.3f ms\n", ms);
  std::printf("bandwidth   : %.1f GB/s (đọc 2 + ghi 1)\n", gbps);
  std::printf("throughput  : %.2f GFLOP/s\n", gflops);
  std::printf("result      : %s (%d sai)\n", wrong == 0 ? "OK" : "FAIL", wrong);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  return wrong == 0 ? 0 : 1;
}
