import re


DANGEROUS_FUNCTIONS = [
    r"\bsystem\s*\(",
    r"\bpopen\s*\(",
    r"\bexec[lvpe]*\s*\(",
    r"\bfork\s*\(",
    r"\bspawn[lp]*\s*\(",
    r"\bunlink\s*\(",
    r"\bremove\s*\(",
    r"\brename\s*\(",
    r"\bmkdir\s*\(",
    r"\brmdir\s*\(",
    r"\bchmod\s*\(",
    r"\bchown\s*\(",
    r"\bkill\s*\(",
    r"\bsignal\s*\(",
    r"\bsocket\s*\(",
    r"\bconnect\s*\(",
    r"\baccept\s*\(",
    r"\bcurl_easy",
    r"\bcurl_",
    r"#include\s*<curl",
    r"#include\s*<sys/socket",
    r"#include\s*<arpa/inet",
    r"#include\s*<netinet",
    r"#include\s*<unistd\.h>",
    r"__asm__?\s*\(",
    r"asm\s+volatile",
    r"asm\s*\(",
]


def scan_code(source_code: str) -> tuple[bool, str]:
    """Scan source code for potentially dangerous constructs.

    Returns (is_safe, error_message).
    """
    if len(source_code) > 65536:
        return False, "代码长度超过限制（最大64KB）"

    lines = source_code.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue

        for pattern in DANGEROUS_FUNCTIONS:
            if re.search(pattern, stripped, re.IGNORECASE):
                return False, f"第{i}行包含不允许的函数调用或头文件：{stripped[:80]}"

    if not re.search(r"__global__\s+void", source_code):
        return False, "未检测到__global__内核函数，请确保代码包含至少一个CUDA kernel"

    if not re.search(r"#include\s*<cuda", source_code) and not re.search(
        r"#include\s*<cuda_runtime", source_code
    ):
        return False, "未检测到CUDA头文件引用"

    return True, ""
