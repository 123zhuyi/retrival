"""AscendCBuilder — bisheng compilation for Ascend C kernel solutions.

Compiles .asc/.cpp/.cc source files using the bisheng compiler.
Runner.pre_build() stages sources, calls build(), then loads .so.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path

import jinja2
import torch
import torch_npu  # noqa: F401 — register NPU device

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import (
    Solution,
    SupportedLanguages,
    SupportedHardware,
    BuildMethod,
)
from npu_kernelbench.builder.builder import Builder, BuildError


_CPP_SOURCE_EXTS = frozenset({".asc", ".cpp", ".cc", ".cxx"})

_NPU_ARCH_MAP: dict[str, str] = {
    "ascend910b": "dav-2201",
    "ascend910_93": "dav-2201",
    "ascend950": "dav-3510",
}

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_template(name: str, **context: object) -> str:
    """Load and render a Jinja2 template from the templates/ directory."""
    template = _jinja_env.get_template(name)
    return template.render(**context)


def _resolve_npu_arch(hw: SupportedHardware) -> str:
    """Resolve --npu-arch value from target hardware."""
    if hw != SupportedHardware.LOCAL:
        arch = _NPU_ARCH_MAP.get(hw.value)
        if arch is None:
            raise BuildError(f"Unknown NPU arch for hardware: {hw.value}")
        return arch

    device_name = torch.npu.get_device_name()
    if device_name.startswith("Ascend950"):
        return "dav-3510"
    if device_name.startswith("Ascend9"):
        return "dav-2201"
    raise BuildError(f"Cannot determine NPU arch from device name: {device_name}")


def _get_bisheng_path() -> str:
    path = shutil.which("bisheng")
    if not path:
        raise BuildError("bisheng compiler not found on PATH")
    return path


def _get_dependency_paths() -> dict[str, list[str]]:
    python_include = sysconfig.get_config_var("INCLUDEPY") or ""
    python_lib = sysconfig.get_config_var("LIBDIR") or ""

    torch_dir = os.path.dirname(torch.__file__)
    torch_include = [
        os.path.join(torch_dir, "include"),
        os.path.join(torch_dir, "include", "torch", "csrc", "api", "include"),
    ]
    torch_lib = os.path.join(torch_dir, "lib")

    torch_npu_dir = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_dir, "include")
    torch_npu_lib = os.path.join(torch_npu_dir, "lib")

    return {
        "includes": [*torch_include, python_include, torch_npu_include],
        "libs": [python_lib, torch_lib, torch_npu_lib],
    }


def _get_catlass_include() -> str | None:
    """Return the catlass include directory from CATLASS_HOME env var.

    Checks $CATLASS_HOME/include first; falls back to $CATLASS_HOME directly.
    Returns None if CATLASS_HOME is not set or the path does not exist.
    """
    catlass_home = os.environ.get("CATLASS_HOME")
    if not catlass_home:
        return None
    include = os.path.join(catlass_home, "include")
    if os.path.isdir(include):
        return include
    if os.path.isdir(catlass_home):
        return catlass_home
    return None


def _get_ascendc_cmake_dir() -> str:
    """Verify ASCEND_HOME_PATH is set (actual path detection is in CMakeLists.txt)."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        raise BuildError("ASCEND_HOME_PATH environment variable is not set")
    return ascend_home


def _compile_with_cmake(
    sources: list[str],
    output_so: Path,
    *,
    abi_value: str,
    npu_arch: str,
    work_dir: Path,
    sources_dir: Path,
    bisheng_cflags: list[str] | None = None,
    ld_flags: list[str] | None = None,
    extra_includes: list[str] | None = None,
    compile_timeout: int = 120,
) -> None:
    """Compile Ascend C kernels via CMake + ascendc.cmake macros."""
    del bisheng_cflags, ld_flags  # not used in cmake flow; reserved for future
    log = logging.getLogger(__name__)
    _get_ascendc_cmake_dir()  # pre-check: ASCEND_HOME_PATH is set

    # Classify source files: op_kernel/ → kernel, rest → host
    kernel_sources = []
    host_sources = []
    for src in sources:
        rel = os.path.relpath(src, work_dir)
        if "op_kernel" in Path(rel).parts:
            kernel_sources.append(rel)
        else:
            host_sources.append(rel)

    dep = _get_dependency_paths()
    include_dirs = [str(work_dir)] + (extra_includes or [])

    soc_version = torch.npu.get_device_name()

    cmake_content = _generate_cmake_lists(
        kernel_sources=kernel_sources,
        host_sources=host_sources,
        include_dirs=list(dict.fromkeys(include_dirs + dep["includes"])),  # dedup, preserve order
        link_dirs=dep["libs"],
        abi_value=abi_value,
        soc_version=soc_version,
    )

    cmake_path = work_dir / "CMakeLists.txt"
    cmake_path.write_text(cmake_content)
    log.info("Generated CMakeLists.txt at %s", cmake_path)

    # Write build.sh for reproducibility (before configure, so it exists even on failure)
    build_sh = work_dir / "build.sh"
    build_sh.write_text(_render_template("build.sh.j2",
        work_dir=str(work_dir),
        abi_value=abi_value,
        soc_version=soc_version,
    ))
    build_sh.chmod(0o755)

    # Configure
    build_dir = work_dir / "build"
    build_dir.mkdir(exist_ok=True)
    configure_cmd = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(work_dir),
        f"-DGLIBCXX_USE_CXX11_ABI={abi_value}",
        f"-DSOC_VERSION={soc_version}",
    ]
    log_path = work_dir / "cmake_compile.log"

    log.info("CMake configure: %s", shlex.join(configure_cmd))
    try:
        result = subprocess.run(configure_cmd, capture_output=True, text=True,
                                cwd=str(work_dir), timeout=compile_timeout)
    except subprocess.TimeoutExpired:
        raise BuildError(
            f"cmake configure timed out after {compile_timeout}s"
        )
    combined = result.stdout + result.stderr
    log_path.write_text(combined, errors="replace")
    if result.returncode != 0:
        raise BuildError(
            f"cmake configure failed (exit {result.returncode}), see {log_path}"
        )

    # Build
    build_cmd = ["cmake", "--build", str(build_dir)]
    log.info("CMake build: %s", shlex.join(build_cmd))
    try:
        result = subprocess.run(build_cmd, capture_output=True, text=True,
                                cwd=str(work_dir), timeout=compile_timeout)
    except subprocess.TimeoutExpired:
        raise BuildError(
            f"cmake build timed out after {compile_timeout}s"
        )
    combined = result.stdout + result.stderr
    with log_path.open("a", errors="replace") as f:
        f.write("\n" + combined)
    if result.returncode != 0:
        raise BuildError(
            f"cmake build failed (exit {result.returncode}), see {log_path}"
        )

    # Locate built .so
    built_so = None
    for so in build_dir.rglob("*.so"):
        built_so = so
        break
    if built_so is None:
        raise BuildError("cmake build completed but no .so file found under build/")
    shutil.copyfile(built_so, output_so)
    log.info("Copied %s -> %s", built_so, output_so)


def _generate_cmake_lists(
    kernel_sources: list[str],
    host_sources: list[str],
    include_dirs: list[str],
    link_dirs: list[str],
    abi_value: str,
    soc_version: str,
) -> str:
    """Generate CMakeLists.txt content for Ascend C kernel compilation."""
    shared_sources = host_sources if host_sources else kernel_sources

    return _render_template("CMakeLists.txt.j2",
        kernel_sources=kernel_sources,
        shared_sources=shared_sources,
        include_dirs=include_dirs,
        link_dirs=link_dirs,
        abi_value=abi_value,
        soc_version=soc_version,
    )


def _compile_with_bisheng(
    sources: list[str],
    output_so: Path,
    *,
    abi_value: str,
    npu_arch: str,
    work_dir: Path,
    bisheng_cflags: list[str] | None = None,
    ld_flags: list[str] | None = None,
    extra_includes: list[str] | None = None,
    compile_timeout: int = 120,
) -> None:
    bisheng = _get_bisheng_path()
    dep = _get_dependency_paths()
    log = logging.getLogger(__name__)

    cmd = [
        bisheng,
        "-x",
        "asc",
        f"--npu-arch={npu_arch}",
        "-shared",
        "-fPIC",
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
        *(bisheng_cflags or []),
        "-ltorch_npu",
        "-ltorch",
        "-lc10",
        *(ld_flags or []),
        *sources,
        "-o",
        str(output_so),
    ]
    for inc in dep["includes"]:
        cmd.append(f"-I{inc}")
    for lib in dep["libs"]:
        cmd.append(f"-L{lib}")
    for inc in extra_includes or []:
        cmd.append(f"-I{inc}")

    cmd_str = shlex.join(cmd)
    log.info("Compilation command: %s", cmd_str)

    build_sh = work_dir / "build.sh"
    build_sh.write_text(f"#!/bin/bash\ncd {shlex.quote(str(work_dir))}\n{cmd_str}\n")
    build_sh.chmod(0o755)

    log_path = work_dir / "bisheng_compile.log"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=compile_timeout)
    except subprocess.TimeoutExpired:
        raise BuildError(
            f"bisheng compilation timed out after {compile_timeout}s"
        )
    combined = result.stdout + result.stderr
    log_path.write_text(combined, errors="replace")
    if result.returncode != 0:
        raise BuildError(
            f"bisheng compilation failed (exit {result.returncode}), see {log_path}"
        )
    if not output_so.exists():
        raise BuildError(f"bisheng did not produce output file: {output_so}")


class AscendCBuilder(Builder):
    """Compiles Ascend C kernels using bisheng."""

    name = "ascendc"

    def can_build(self, solution: Solution) -> bool:
        return (
            SupportedLanguages.ASCENDC in solution.spec.languages
            or SupportedLanguages.CATLASS in solution.spec.languages
        )

    def build(
        self,
        definition: Definition,
        solution: Solution,
        work_dir: str = "/tmp/npu-kernelbench",
        compile_timeout: int = 120,
    ) -> None:
        spec = solution.spec
        work_path = Path(work_dir)
        sources_dir = work_path / "sources"

        cpp_sources = [
            s for s in solution.sources if Path(s.path).suffix in _CPP_SOURCE_EXTS
        ]
        if not cpp_sources:
            raise BuildError("No compilable source files found", builder_name=self.name)

        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)
        log.info("Work directory: %s", work_path)
        log.info("Source files: %s", [s.path for s in cpp_sources])

        output_so = work_path / "kernel.so"
        abi_value = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
        source_paths = [str(sources_dir / s.path) for s in cpp_sources]

        if len(spec.target_hardware) != 1:
            raise BuildError(
                f"AscendC solutions require exactly 1 target_hardware, "
                f"got {len(spec.target_hardware)}",
                builder_name=self.name,
            )
        npu_arch = _resolve_npu_arch(spec.target_hardware[0])

        extra_includes = [str(sources_dir)]
        for root, dirs, files in os.walk(sources_dir):
            if any(f.endswith((".h", ".hpp")) for f in files):
                extra_includes.append(root)

        if SupportedLanguages.CATLASS in spec.languages:
            catlass_inc = _get_catlass_include()
            if catlass_inc:
                extra_includes.append(catlass_inc)
                log.info("Catlass include path: %s", catlass_inc)
            else:
                log.warning(
                    "CATLASS language specified but CATLASS_HOME is not set "
                    "or path does not exist"
                )

        if spec.build_method == BuildMethod.CMAKE:
            _compile_with_cmake(
                source_paths,
                output_so,
                abi_value=abi_value,
                npu_arch=npu_arch,
                work_dir=work_path,
                sources_dir=sources_dir,
                extra_includes=extra_includes,
                compile_timeout=compile_timeout,
            )
        else:
            _compile_with_bisheng(
                source_paths,
                output_so,
                abi_value=abi_value,
                npu_arch=npu_arch,
                work_dir=work_path,
                bisheng_cflags=spec.bisheng_cflags,
                ld_flags=spec.ld_flags,
                extra_includes=extra_includes,
                compile_timeout=compile_timeout,
            )
        log.info("Compilation succeeded: %s", output_so)
