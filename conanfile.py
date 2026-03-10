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
        self.requires("pybind11/2.13.6")
        if self.options.with_conan_python:
            self.requires("tttapa-python-dev/3.13.7")

    def layout(self):
        cmake_layout(self)
