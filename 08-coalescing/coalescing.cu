#include <cuda_runtime.h>

#include <cstdlib>
__global__ void Access(const float* in, float* out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int j = (i * stride) & (n - 1);
    out[i] = in[j] + 1.0f;
  }
}
int main(int argc, char** argv) {
  int stride = argc > 1 ? std::atoi(argv[1]) : 1;
  constexpr int kN = 1 << 26;  // Power of two for cheap masking.
  float *a = nullptr, *b = nullptr;
  cudaMalloc(&a, kN * sizeof(float));
  cudaMalloc(&b, kN * sizeof(float));
  dim3 block(256), grid((kN + block.x - 1) / block.x);
  for (int i = 0; i < 8; ++i) Access<<<grid, block>>>(a, b, kN, stride);
  cudaDeviceSynchronize();
  cudaFree(a);
  cudaFree(b);
}