#include <cuda_runtime.h>

#include <cstdlib>
__global__ void Branch(float* x, int n, bool divergent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bool take_a =
      divergent ? ((threadIdx.x & 1) == 0) : (((threadIdx.x / 32) & 1) == 0);
  float v = x[i];
  if (take_a) {
#pragma unroll 32
    for (int k = 0; k < 32; ++k) v = v * 1.0001f + 1.0f;
  } else {
#pragma unroll 32
    for (int k = 0; k < 32; ++k) v = v * 0.9999f - 1.0f;
  }
  x[i] = v;
}
int main(int argc, char** argv) {
  bool divergent = argc > 1 && std::atoi(argv[1]);
  constexpr int kN = 1 << 24;
  float* d = nullptr;
  cudaMalloc(&d, kN * sizeof(float));
  for (int i = 0; i < 8; ++i) {
    Branch<<<(kN + 255) / 256, 256>>>(d, kN, divergent);
  }
  cudaDeviceSynchronize();
  cudaFree(d);
}