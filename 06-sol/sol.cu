#include <cuda_runtime.h>

#include <cstdlib>
__global__ void Saxpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
int main() {
  constexpr int kN = 1 << 26;
  float *x = nullptr, *y = nullptr;
  cudaMalloc(&x, kN * sizeof(float));
  cudaMalloc(&y, kN * sizeof(float));
  dim3 block(256), grid((kN + block.x - 1) / block.x);
  for (int i = 0; i < 20; ++i) {
    Saxpy<<<grid, block>>>(y, x, 2.0f, kN);
  }
  cudaDeviceSynchronize();
  cudaFree(x);
  cudaFree(y);
}