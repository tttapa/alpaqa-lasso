#include <cmath>

__global__ void l1_prox_grad_step_kernel(const double *x, const double *grad,
                                         double *out, double *step, size_t size,
                                         double stepsize, double lambda) {
    auto tid = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
    if (tid < size) {
        double x_fw   = x[tid] - stepsize * grad[tid];
        double γλ     = stepsize * lambda;
        double result = std::fmin(std::fmax(0., x_fw - γλ), x_fw + γλ);
        out[tid]      = result;
        step[tid]     = result - x[tid];
    }
}

void l1_prox_grad_step_gpu(const double *x, const double *grad, double *out,
                           double *step, size_t size, double stepsize,
                           double lambda, cudaStream_t stream) {
    auto blockSize = 256u;
    auto gridSize  = static_cast<unsigned>((size + blockSize - 1) / blockSize);
    l1_prox_grad_step_kernel<<<gridSize, blockSize, 0, stream>>>(
        x, grad, out, step, size, stepsize, lambda);
}
