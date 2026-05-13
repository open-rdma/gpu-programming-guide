import os
import subprocess
import tempfile

COMPILE_TIMEOUT = int(os.environ.get("COMPILE_TIMEOUT", "60"))


def compile_cuda(source_code: str, arch: str = "sm_86") -> tuple[bool, str, str | None]:
    """Compile CUDA source code with nvcc.

    Args:
        source_code: The CUDA source code string.
        arch: Target compute capability (default sm_86 for RTX 3060).

    Returns:
        (success, output, binary_path_or_error)
    """
    tmpdir = tempfile.mkdtemp(prefix="cuda_compile_")
    src_path = os.path.join(tmpdir, "kernel.cu")
    out_path = os.path.join(tmpdir, "kernel")

    try:
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        result = subprocess.run(
            [
                "nvcc",
                "-arch=" + arch,
                "-o", out_path,
                src_path,
                "-lineinfo",
            ],
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT,
        )

        if result.returncode == 0:
            return True, result.stdout + result.stderr, out_path
        return False, result.stdout + result.stderr, None

    except subprocess.TimeoutExpired:
        return False, f"编译超时（{COMPILE_TIMEOUT}秒）", None
    except FileNotFoundError:
        return False, "nvcc未找到，请确认CUDA工具包已安装", None
    except Exception as e:
        return False, f"编译过程出错：{str(e)}", None
