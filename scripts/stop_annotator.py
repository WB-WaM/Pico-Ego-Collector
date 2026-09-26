#!/usr/bin/env python3
"""Release this project's old annotator, including job-control-stopped processes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import time


def process_identity(pid: int) -> tuple[str, str] | None:
    try:
        # comm may contain spaces or parentheses; the last ')' ends that field.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]  # state, starttime (Linux stat field 22)
    except (OSError, IndexError):
        return None


def annotator_processes(root: Path, port: int) -> list[tuple[int, str]]:
    expected_script = (root / "scripts/serve_annotator.py").resolve()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [os.fsdecode(s) for s in (proc / "cmdline").read_bytes().split(b"\0") if s]
            cwd = (proc / "cwd").resolve(strict=True)
            for i, arg in enumerate(argv[1:], 1):
                if Path(arg).name != "serve_annotator.py":
                    continue
                if (cwd / arg).resolve() != expected_script:
                    continue
                server_port = 8765
                for j in range(i + 1, len(argv)):
                    if argv[j] == "--port":
                        server_port = int(argv[j + 1])
                    elif argv[j].startswith("--port="):
                        server_port = int(argv[j].split("=", 1)[1])
                identity = process_identity(int(proc.name))
                if server_port == port and identity and identity[0] != "Z":
                    found.append((int(proc.name), identity[1]))
                break
        except (OSError, ValueError, IndexError):
            continue
    return found


def still_running(process: tuple[int, str]) -> bool:
    current = process_identity(process[0])
    return current is not None and current[1] == process[1] and current[0] != "Z"


def send(process: tuple[int, str], sig: int) -> None:
    if still_running(process):
        try:
            os.kill(process[0], sig)
        except ProcessLookupError:
            pass


def wait_for_exit(processes: list[tuple[int, str]], timeout: float) -> list[tuple[int, str]]:
    deadline = time.monotonic() + timeout
    while True:
        processes = [p for p in processes if still_running(p)]
        if not processes or time.monotonic() >= deadline:
            return processes
        time.sleep(0.05)


def stop_previous(root: Path, host: str, port: int, timeout: float = 3.0) -> None:
    previous = annotator_processes(root.resolve(), port)
    if previous:
        print("Stopping previous annotator processes: " + " ".join(str(p[0]) for p in previous), flush=True)
    for process in previous:
        send(process, signal.SIGTERM)
        # SIGTERM stays pending while a process is stopped (for example Ctrl-Z).
        # Resume that same process so it can terminate and release its socket.
        send(process, signal.SIGCONT)
    remaining = wait_for_exit(previous, timeout)
    for process in remaining:
        print(f"Process {process[0]} did not stop; terminating it.", flush=True)
        send(process, signal.SIGKILL)
    if wait_for_exit(remaining, 1.0):
        raise RuntimeError("Previous server is still running. New server was not started.")
    # Use the same address family/reuse policy as ThreadingHTTPServer. An
    # unrelated listener is reported, never killed merely for sharing a port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Cannot bind {host}:{port}: {exc}. Check which process uses the port, or set PICO_WEB_PORT.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        stop_previous(args.project_root, args.host, args.port)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
