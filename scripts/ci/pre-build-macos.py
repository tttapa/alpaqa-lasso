import os
import re
import platform
from pathlib import Path
from subprocess import run

os.chdir(Path(__file__).parent.parent.parent)

archflags = os.getenv("ARCHFLAGS")
archs = ()
if archflags:
    archs = tuple(sorted(re.findall(r"-arch +(\S+)", archflags)))

print("ARCHFLAGS:", archs)
print("PLATFORM: ", platform.machine())
print("MACOSX_DEPLOYMENT_TARGET:", os.getenv("MACOSX_DEPLOYMENT_TARGET"))
print(flush=True)

deployment_target = os.getenv("MACOSX_DEPLOYMENT_TARGET", "10.9")
can_run = platform.machine() in archs
conan_arch = {
    (): "ci-native",
    ("x86_64",): "ci-x86_64",
    ("arm64",): "ci-arm64",
    ("arm64", "x86_64"): "ci-universal2",
}[archs]
cpu_flags = {
    (): (),
    ("x86_64",): ("-march=skylake",),
    ("arm64",): ("-mcpu=apple-m1",),
    ("arm64", "x86_64"): (),
}[archs]

native_profile = f"""\
include(default)
[settings]
arch={conan_arch}
os.version={deployment_target}
build_type=Release
[conf]
tools.cmake.cmaketoolchain:generator=Ninja Multi-Config
tools.build:skip_test=True
"""

cross_profile = f"""\
include(default)
[settings]
arch={conan_arch}
os.version={deployment_target}
build_type=Release
[conf]
tools.cmake.cmaketoolchain:user_toolchain=["{{{{ os.path.join(profile_dir, "cibw.toolchain") }}}}"]
tools.build.cross_building:can_run={can_run}
tools.cmake.cmaketoolchain:generator=Ninja Multi-Config
tools.build:skip_test=True
"""

cross_toolchain = f"""\
set(CMAKE_SYSTEM_NAME "Darwin")
set(CMAKE_OSX_ARCHITECTURES  "{';'.join(archs)}")
set(CMAKE_C_FLAGS_INIT       "{' '.join(cpu_flags)}")
set(CMAKE_CXX_FLAGS_INIT     "{' '.join(cpu_flags)}")
set(CMAKE_Fortran_FLAGS_INIT "{' '.join(cpu_flags)}")
"""

Path("cibw.profile").write_text(cross_profile if archs else native_profile)
if archs:
    Path("cibw.toolchain").write_text(cross_toolchain)

opts = dict(shell=True, check=True)
if os.path.isdir("alpaqa"):
    run("conan create ./alpaqa -pr:h ./cibw.profile --build=missing", **opts)
run("conan install . -pr:h ./cibw.profile --build=missing", **opts)
