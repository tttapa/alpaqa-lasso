#include <complex-openmp/complex-convert.hpp>
#include <complex-openmp/complex-omp-problem.hpp>

#include <alpaqa/util/io/csv.hpp>
#include <fstream>

#if WITH_PYTHON
#include <pybind11/eigen/tensor.h>
#endif

namespace acl {

void ComplexOMPProblem::load_data(fs::path csv_file) {
    data_file = std::move(csv_file);
    std::ifstream ifile{data_file};
    if (!ifile)
        throw std::runtime_error("Unable to open file '" + data_file.string() +
                                 "'");
    // Load dimensions (#observations, #features, #targets, #terms)
    auto dims = alpaqa::csv::read_row_std_vector<length_t>(ifile);
    if (dims.size() < 2 || dims.size() > 4)
        throw std::runtime_error("Invalid problem dimensions in data file \'" +
                                 data_file.string() + '\'');
    m = dims[0];
    n = dims[1];
    p = dims.size() > 2 ? dims[2] : 1;
    q = dims.size() > 3 ? dims[3] : 1;
    storage.b.resize(m, p * q);
    storage.A.resize(m, n * q);
    // Read the measurements
    for (length_t i = 0; i < p * q; ++i)
        alpaqa::csv::read_row(ifile, c2r(rcmat{storage.b.col(i)}));
    // Read the data
    for (length_t i = 0; i < n * q; ++i)
        alpaqa::csv::read_row(ifile, c2r(rcmat{storage.A.col(i)}));
    data.A.emplace(crcmat{storage.A});
    data.b.emplace(crcmat{storage.b});
}

#if WITH_PYTHON
void ComplexOMPProblem::load_data(py::kwargs kwargs) {
    using cmctensor3 = Eigen::TensorMap<const Eigen::Tensor<cplx_t, 3>>;
    if (kwargs.contains("lambda_1"))
        λ_1 = py::cast<real_t>(kwargs.attr("pop")("lambda_1"));
    if (kwargs.contains("lambda_2"))
        λ_2 = py::cast<real_t>(kwargs.attr("pop")("lambda_2"));
    py_storage.A  = kwargs.attr("pop")("A");
    py_storage.b  = kwargs.attr("pop")("b");
    auto A_tensor = py::cast<cmctensor3>(py_storage.A);
    auto b_tensor = py::cast<cmctensor3>(py_storage.b);
    n             = A_tensor.dimension(1);
    m             = A_tensor.dimension(0);
    p             = b_tensor.dimension(1);
    q             = A_tensor.dimension(2);
    if (m != b_tensor.dimension(0))
        throw std::invalid_argument("Number of rows of A and b should match");
    if (q != b_tensor.dimension(2))
        throw std::invalid_argument("Batch size of A and b should match");
    data.A.emplace(cmcmat{A_tensor.data(), m, n * q});
    data.b.emplace(cmcmat{b_tensor.data(), m, p * q});
}
#endif

void ComplexOMPProblem::init() {
    loss_scale = 1 / static_cast<real_t>(m);
    work.Ax.resize(m, p * q);
}

} // namespace acl