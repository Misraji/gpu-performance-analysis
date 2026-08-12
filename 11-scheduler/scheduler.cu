#include <cuda_runtime.h>

#include <cstdlib>
__global__ void Arithmetic(float* x, int n, bool dependent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = x[i], b = a + 1, c = a + 2, d = a + 3;
  for (int k = 0; k < 256; ++k) {
    if (dependent) {
      a = a * 1.000001f + 0.1f;  // One chain.
    } else {
      a = a * 1.000001f + 0.1f;
      b = b * 1.000002f + 0.2f;
      c = c * 1.000003f + 0.3f;
      d = d * 1.000004f + 0.4f;  // Independent chains.
    }
  }
  x[i] = a + b + c + d;
}
int main(int argc, char** argv) {
  bool dep = argc > 1 && std::atoi(argv[1]);
  constexpr int kN = 1 << 22;
  float* d = nullptr;
  cudaMalloc(&d, kN * sizeof(float));
  for (int i = 0; i < 8; ++i) {
    Arithmetic<<<(kN + 255) / 256, 256>>>(d, kN, dep);
  }
  cudaDeviceSynchronize();
  cudaFree(d);
}
