"""Hardware and environment detection for the node agent."""
from __future__ import annotations

import platform
import shutil
import subprocess


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def detect() -> dict:
    system = platform.system()
    info = {'os': f'{system} {platform.release()}', 'chip': platform.machine(), 'total_mem_bytes': 0, 'gpus': []}
    if system == 'Darwin':
        info['chip'] = _run('sysctl', '-n', 'machdep.cpu.brand_string') or info['chip']
        info['total_mem_bytes'] = int(_run('sysctl', '-n', 'hw.memsize') or 0)
        gpu = _run('system_profiler', 'SPDisplaysDataType')
        if gpu:
            info['gpus'] = [l.split(':', 1)[1].strip() for l in gpu.splitlines() if 'Chipset Model' in l]
    else:
        try:
            with open('/proc/meminfo') as fh:
                kb = next(int(l.split()[1]) for l in fh if l.startswith('MemTotal'))
            info['total_mem_bytes'] = kb * 1024
        except (OSError, StopIteration):
            pass
        smi = _run('nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader') if shutil.which('nvidia-smi') else None
        if smi:
            info['gpus'] = [l.strip() for l in smi.splitlines()]
            info['chip'] = info['gpus'][0].split(',')[0]
        if 'microsoft' in platform.release().lower():
            info['os'] += ' (WSL2)'
    return info


def tailscale_ip() -> str | None:
    out = _run('tailscale', 'ip', '-4')
    return out.splitlines()[0].strip() if out else None


def supports_rpc(rpc_server: str) -> bool:
    out = _run(rpc_server, '--help')
    return bool(out) or shutil.which(rpc_server) is not None
