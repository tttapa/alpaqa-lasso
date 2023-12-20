#include <cuda/cuda-problem.hpp>
#include <stdexcept>

#if WITH_PYTHON
#include <pybind11/numpy.h>
#endif

namespace acl {

void CUDAProblem::load_data([[maybe_unused]] fs::path csv_file) {
    throw std::logic_error("CSV loading not yet implemented");
}

#if WITH_PYTHON

template <ptrdiff_t Rank>
auto load_tensor(const py::object &o) {
    if (!py::isinstance<py::array>(o))
        throw std::logic_error("Invalid type: expected array");
    auto arr = py::reinterpret_borrow<py::array>(o);
    if ((arr.flags() & py::array::f_style) == 0)
        throw std::logic_error("Invalid storage order: expected Fortran style");
    if (!arr.dtype().is(py::dtype::of<real_t>()))
        throw std::logic_error("Invalid element type: expected float64");
    if (arr.ndim() != Rank)
        throw std::logic_error("Invalid rank: expected " +
                               std::to_string(Rank));
    struct {
        const real_t *data;
        std::array<length_t, Rank> shape;
        length_t size;
    } result;
    result.data = static_cast<const real_t *>(arr.data());
    std::copy(arr.shape(), arr.shape() + arr.ndim(), result.shape.begin());
    result.size = arr.size();
    return result;
}

void CUDAProblem::load_data(py::kwargs kwargs) {
    auto A = load_tensor<3>(kwargs["A"]);
    auto b = load_tensor<3>(kwargs["b"]);
    if (A.shape[0] != b.shape[0])
        throw std::logic_error("Mismatching rows of A and b");
    if (A.shape[2] != b.shape[2])
        throw std::logic_error("Mismatching batch sizes of A and b");
    m = A.shape[0];
    n = A.shape[1];
    p = b.shape[1];
    q = A.shape[2];
    assert(A.size == (m * n * q));
    assert(b.size == (m * p * q));
    gpu.data = {
        .A = acl::cudaAlloc<real_t>(static_cast<size_t>(m * n * q)),
        .b = acl::cudaAlloc<real_t>(static_cast<size_t>(m * p * q)),
    };
    auto stream1 = acl::cudaStreamAlloc(cudaStreamNonBlocking),
         stream2 = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    check(cudaMemcpyAsync(gpu.data.A.get(), A.data,
                          sizeof(real_t) * static_cast<size_t>(m * n * q),
                          cudaMemcpyHostToDevice, stream1.get()),
          "memcpy A");
    check(cudaMemcpyAsync(gpu.data.b.get(), b.data,
                          sizeof(real_t) * static_cast<size_t>(m * p * q),
                          cudaMemcpyHostToDevice, stream2.get()),
          "memcpy b");
    cudaStreamSynchronize(stream1.get());
    cudaStreamSynchronize(stream2.get());
}

#endif

void CUDAProblem::init() {
    handle        = acl::cublasUniqueCreate();
    gpu.constants = {
        .zeros      = acl::cudaAlloc<real_t>(static_cast<size_t>(q)),
        .ones       = acl::cudaAlloc<real_t>(static_cast<size_t>(q)),
        .minus_ones = acl::cudaAlloc<real_t>(static_cast<size_t>(q)),
        .minus_twos = acl::cudaAlloc<real_t>(static_cast<size_t>(q)),
        .loss_scale = acl::cudaAlloc<real_t>(static_cast<size_t>(q)),
        .λ_2        = acl::cudaAlloc<real_t>(static_cast<size_t>(q)),
    };
    gpu.work = {
        .x     = acl::cudaAlloc<real_t>(static_cast<size_t>(n * p * q)),
        .Ax    = acl::cudaAlloc<real_t>(static_cast<size_t>(m * p * q)),
        .norms = acl::cudaAlloc<real_t>(2),
    };
    auto copy_constant_to_gpu = [work = vec(q)](real_t value,
                                                real_t *dst) mutable {
        work.setConstant(value);
        check(cublasSetVector(static_cast<int>(work.size()), sizeof(real_t),
                              work.data(), 1, dst, 1),
              "copy_constant_to_gpu");
    };
    copy_constant_to_gpu(0, gpu.constants.zeros.get());
    copy_constant_to_gpu(1, gpu.constants.ones.get());
    copy_constant_to_gpu(-1, gpu.constants.minus_ones.get());
    copy_constant_to_gpu(-2, gpu.constants.minus_twos.get());
    copy_constant_to_gpu(loss_scale, gpu.constants.loss_scale.get());
    copy_constant_to_gpu(λ_2, gpu.constants.λ_2.get());
}

#if WITH_PYTHON
py::object CUDAProblem::set_λ_2(py::args args, py::kwargs kwargs) {
    auto ret = Problem::set_λ_2(std::move(args), std::move(kwargs));
    vec work = vec::Constant(q, λ_2);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          gpu.constants.λ_2.get(), 1),
          "set λ_2");
    return ret;
}
#endif

} // namespace acl