#include <cuda_runtime.h>
__global__ void ManyRegisters(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a0 = x[i], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4,
        a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
#pragma unroll 128
  for (int k = 0; k < 128; ++k) {
    a0 = a0 * 1.00001f + a4;
    a1 = a1 * 1.00002f + a5;
    a2 = a2 * 1.00003f + a6;
    a3 = a3 * 1.00004f + a7;
    a4 = a4 * 0.99999f + a0;
    a5 = a5 * 0.99998f + a1;
    a6 = a6 * 0.99997f + a2;
    a7 = a7 * 0.99996f + a3;
  }
  x[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
int main() {
  constexpr int n = 1 << 22;
  float* d = nullptr;
  cudaMalloc(&d, n * sizeof(float));
  ManyRegisters<<<(n + 255) / 256, 256>>>(d, n);
  cudaDeviceSynchronize();
  cudaFree(d);
}