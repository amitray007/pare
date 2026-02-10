import asyncio

from config import settings
from exceptions import ToolTimeoutError, OptimizationError


async def run_tool(
    cmd: list[str],
    input_data: bytes,
    timeout: int | None = None,
    allowed_exit_codes: set[int] | None = None,
) -> tuple[bytes, bytes, int]:
    """Run a CLI tool with stdin/stdout piping.

    All compression tools are invoked this way — bytes in via stdin,
    bytes out via stdout. No temp files, no disk I/O.

    Args:
        cmd: Command and arguments (e.g., ["pngquant", "--quality", "65-80", "-"]).
        input_data: Raw bytes to pipe to stdin.
        timeout: Seconds before killing the process.
            Defaults to settings.tool_timeout_seconds.
        allowed_exit_codes: Exit codes that are not errors (besides 0).
            E.g., {99} for pngquant's "quality too low" code.

    Returns:
        Tuple of (stdout_bytes, stderr_bytes, return_code).

    Raises:
        ToolTimeoutError: If the process exceeds the timeout.
        OptimizationError: If the process exits with an unexpected non-zero code.
    """
    if timeout is None:
        timeout = settings.tool_timeout_seconds
    if allowed_exit_codes is None:
        allowed_exit_codes = set()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_data),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolTimeoutError(
            f"Tool {cmd[0]} timed out after {timeout}s",
            tool=cmd[0],
            timeout=timeout,
        )

    if proc.returncode != 0 and proc.returncode not in allowed_exit_codes:
        raise OptimizationError(
            f"{cmd[0]} failed with exit code {proc.returncode}: "
            f"{stderr.decode(errors='replace')[:500]}",
            tool=cmd[0],
            exit_code=proc.returncode,
        )

    return stdout, stderr, proc.returncode
