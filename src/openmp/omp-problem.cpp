#include <openmp/omp-problem.hpp>

#include <alpaqa/util/io/csv.hpp>
#include <fstream>

#if WITH_PYTHON
#include <pybind11/eigen/tensor.h>
#endif

namespace acl {

void OMPProblem::load_data(fs::path csv_file) {
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
    storage.b.resize(m, p, q);
    storage.A.resize(m, n, q);
    // Read the measurements
    for (length_t i = 0; i < q; ++i)
        for (length_t j = 0; j < p; ++j)
            alpaqa::csv::read_row(
                ifile, mvec{storage.b.data() + j * m + i * m * p, m});
    // Read the data
    for (length_t i = 0; i < q; ++i)
        for (length_t j = 0; j < n; ++j)
            alpaqa::csv::read_row(
                ifile, mvec{storage.A.data() + j * m + i * m * n, m});
    data.A = storage.A;
    data.b = storage.b;
}

#if WITH_PYTHON
void OMPProblem::load_data(py::kwargs kwargs) {
    py_storage.A = kwargs["A"];
    py_storage.b = kwargs["b"];
    data.A       = py::cast<cmtensor3>(py_storage.A);
    data.b       = py::cast<cmtensor3>(py_storage.b);
    n            = data.A.dimension(1);
    m            = data.A.dimension(0);
    p            = data.b.dimension(1);
    q            = data.b.dimension(2);
    if (m != data.b.dimension(0))
        throw std::invalid_argument("Number of rows of A and b should match");
    if (q != data.b.dimension(2))
        throw std::invalid_argument("Batch size of A and b should match");
}
#endif

void OMPProblem::init() {
    loss_scale = 1 / static_cast<real_t>(m);
    work.Ax.resize(m, p, q);
}

} // namespace acl