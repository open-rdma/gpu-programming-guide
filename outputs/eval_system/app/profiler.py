import os
import subprocess
import uuid

PROFILE_TIMEOUT = int(os.environ.get("PROFILE_TIMEOUT", "300"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", "results")


def run_ncu_profile(binary_path: str) -> tuple[bool, str, str | None]:
    """Run NCU profiling on a compiled CUDA binary.

    Returns:
        (success, summary_text, ncu_report_path_or_error)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report_name = f"{uuid.uuid4().hex}.ncu-rep"
    report_path = os.path.join(RESULTS_DIR, report_name)

    try:
        result = subprocess.run(
            [
                "ncu",
                "--set", "full",
                "-o", report_path,
                "--import-source", "yes",
                binary_path,
            ],
            capture_output=True,
            text=True,
            timeout=PROFILE_TIMEOUT,
        )

        summary = _extract_summary(result.stdout + result.stderr)

        if os.path.exists(report_path):
            return True, summary, report_name
        return False, result.stdout + result.stderr, None

    except subprocess.TimeoutExpired:
        return False, f"评测超时（{PROFILE_TIMEOUT}秒）", None
    except FileNotFoundError:
        return False, "ncu未找到，请确认CUDA工具包已安装", None
    except Exception as e:
        return False, f"评测过程出错：{str(e)}", None


def _extract_summary(ncu_output: str) -> str:
    """Extract key performance metrics from NCU output."""
    lines = ncu_output.split("\n")
    important_lines = []
    capture = False

    for line in lines:
        if "==PROF==" in line or "==ERROR==" in line:
            capture = True
            important_lines.append(line.strip())
        elif capture and line.strip():
            important_lines.append(line.strip())
        elif capture and not line.strip():
            capture = False
        if len(important_lines) > 100:
            break

    if not important_lines:
        important_lines = lines[-50:]

    return "\n".join(important_lines)
