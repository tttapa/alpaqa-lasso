#pragma once

#include <openmp/omp-problem.hpp>

namespace acl::util {

real_t normSquared(const crtensor2 &t) {
    return cmvec{t.data(), t.size()}.squaredNorm();
}

real_t dot(const crtensor2 &t, const crtensor2 &s) {
    return cmvec{t.data(), t.size()}.dot(cmvec{s.data(), s.size()});
}

} // namespace acl::util