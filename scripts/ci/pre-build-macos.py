import os
import re
import platform
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)

archflags = os.getenv("ARCHFLAGS")
archs = ()
if archflags:
    archs = tuple(sorted(re.findall(r"-arch +(\S+)", archflags)))
print("ARCHFLAGS:", archs)
print("PLATFORM: ", platform.machine())

can_run = platform.machine() in archs
conan_arch = {
    (): None,
    ("x86_64",): "x86_64",
    ("arm64",): "armv8.3",
    ("arm64", "x86_64"): "avr",  # Just to have a distinct architecture
}[archs]

native_profile = """\
include(default)
[settings]
build_type=Release
[conf]
tools.cmake.cmaketoolchain:generator=Ninja Multi-Config
tools.build:skip_test=True
"""

cross_profile = f"""\
include(default)
[settings]
arch={conan_arch}
build_type=Release
[conf]
tools.cmake.cmaketoolchain:user_toolchain=["{{{{ os.path.join(profile_dir, "cibw.toolchain") }}}}"]
tools.build.cross_building:can_run={can_run}
tools.cmake.cmaketoolchain:generator=Ninja Multi-Config
tools.build:skip_test=True
"""

cross_toolchain = f"""\
set(CMAKE_SYSTEM_NAME "Darwin")
set(CMAKE_OSX_ARCHITECTURES "{';'.join(archs)}")
"""

Path("cibw.profile").write_text(cross_profile if archs else native_profile)
if archs:
    Path("cibw.toolchain").write_text(cross_toolchain)

if os.path.isdir("alpaqa"):
    os.system("conan create ./alpaqa -pr:h ./cibw.profile")
os.system("conan install . -pr:h ./cibw.profile --build=missing")
