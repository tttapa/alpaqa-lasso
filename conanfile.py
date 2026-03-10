from conan import ConanFile
from conan.tools.cmake import cmake_layout


class AlpaqaLassoConan(ConanFile):
    name = "alpaqa-lasso"
    version = "1.1.0-alpha.1"

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "with_conan_python": [True, False],
    }
    default_options = {
        "with_conan_python": False,
    }

    generators = "CMakeDeps", "CMakeToolchain"

    def configure(self):
        self.requires("alpaqa/1.1.0-alpha.1")
        # Note: in theory, this should match the pybind11 version that alpaqa
        # was built with (which is 2.13.6 for alpaqa 1.1.0-alpha.1), to avoid
        # ABI issues. However, importing numpy from 2.13.6 causes segfaults in
        # import_numpy_core_submodule on older versions of Python (e.g. 3.10),
        # so we use a newer version of pybind11 that doesn't have this issue
        # (and we cross our fingers that there are no ABI issues).
        # Future versions of alpaqa will also use newer versions of pybind11,
        # so this should be fine in the long run.
        self.requires("pybind11/3.0.1")
        if self.options.with_conan_python:
            self.requires("tttapa-python-dev/3.13.7")

    def layout(self):
        cmake_layout(self)
