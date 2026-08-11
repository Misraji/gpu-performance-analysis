#include <cuda_runtime.h>

#include <cstdlib>
__global__ void Work(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
#pragma unroll 16
    for (int k = 0; k < 16; ++k) {
      v = v * 1.000001f + 0.000001f;
    }
    x[i] = v;
  }
}
int main(int argc, char** argv) {
  const bool sync_each = argc < 2 || std::atoi(argv[1]) == 0;
  constexpr int kN = 1 << 22;
  constexpr int kIterations = 100;
  float* d = nullptr;
  cudaMalloc(&d, kN * sizeof(float));
  dim3 block(256), grid((kN + block.x - 1) / block.x);
  for (int i = 0; i < kIterations; ++i) {
    Work<<<grid, block>>>(d, kN);
    if (sync_each) {
      cudaDeviceSynchronize();
    }
  }
  cudaDeviceSynchronize();
  cudaFree(d);
}