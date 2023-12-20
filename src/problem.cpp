#include <problem.hpp>

#include <stdexcept>
#include <string>

#if WITH_PYTHON
#include <pybind11/eigen/tensor.h>
#endif

namespace acl {

std::string Problem::get_name() const {
    if (!data_file.empty())
        return "alpaqa-lasso ('" + data_file.string() + "', " +
               std::to_string(m) + "×" + std::to_string(n) + "×" +
               std::to_string(p) + "×" + std::to_string(q) + ")";
    else
        return "alpaqa-lasso (NumPy, " + std::to_string(m) + "×" +
               std::to_string(n) + "×" + std::to_string(p) + "×" +
               std::to_string(q) + ")";
}

#if WITH_PYTHON
py::object Problem::set_λ_1(py::args args, py::kwargs kwargs) {
    if (args.size() != 1)
        throw std::invalid_argument("Invalid number of positional arguments");
    if (!kwargs.empty())
        throw std::invalid_argument("Unexpected keyword arguments");
    return py::cast(std::exchange(λ_1, py::cast<real_t>(args[0])));
}
py::object Problem::set_λ_2(py::args args, py::kwargs kwargs) {
    if (args.size() != 1)
        throw std::invalid_argument("Invalid number of positional arguments");
    if (!kwargs.empty())
        throw std::invalid_argument("Unexpected keyword arguments");
    return py::cast(std::exchange(λ_2, py::cast<real_t>(args[0])));
}
#endif

} // namespace acl
