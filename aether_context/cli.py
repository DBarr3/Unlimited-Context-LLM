"""``aether-context`` console script — init / pool-resize / doctor / bench.

Thin by design: every command is a few lines that call into the library. The CLI never
contains engine logic; it wires argparse to :mod:`aether_context.config`, the Ollama probe in
:mod:`aether_context.local_llm`, and the bench script.

Commands
--------
``aether-context init [--pool N] [--dir D]``
    Initialize / re-initialize the pool config. **Non-tty safe** (build plan §12 CRITICAL):
    the interactive slider only runs when stdin is a real tty; otherwise the size comes from
    ``--pool N`` (>=5), then ``$AETHER_POOL_GB``, then the 5 GB default — it never blocks.

``aether-context --pool N [--dir D]``
    Resize the pool (non-destructive re-index): rewrite ``pool_gb`` in the persisted config
    without touching the on-disk payloads. Rejects ``N < 5`` with the floor reason.

``aether-context doctor [--model M] [--dir D]``
    Runs **fully offline**. Checks the three things that ever go wrong (docs/local-models.md):
    Ollama reachable? model pulled? free RAM vs the configured index? — and prints the *exact*
    fix command for each. Never raises on a down daemon; reports it as a fixable condition.

``aether-context bench [--quick] [--model M] [--json]``
    Delegates to ``bench/drift_vs_window.py`` (engine ON vs OFF). Hermetic by default.

No ``print`` discipline note: this is the *one* place user-facing text is intentional, so the
CLI uses ``print`` for its report. The library proper (everything under ``aether_context`` that
is imported by ``Session``) stays silent and uses the logging seam.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Sequence

from aether_context import __version__
from aether_context.config import (
    POOL_GB_FLOOR,
    PoolConfig,
    reach_tokens,
)
from aether_context.errors import AetherContextError

#: Environment variable read for the pool size when no ``--pool`` flag is given (non-tty).
_ENV_POOL_GB = "AETHER_POOL_GB"
#: Environment variable for the Ollama host the doctor probes (overrides the default).
_ENV_OLLAMA_HOST = "OLLAMA_HOST"
#: Default Ollama host (mirrors local_llm.DEFAULT_OLLAMA_HOST without importing the adapter).
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
#: In-RAM ANN index size per GB of pool (MB). From the README table: 5 GB -> ~145 MB,
#: 10 GB -> ~291 MB, ... i.e. ~29 MB of resident index per GB of reach (the HNSW graph +
#: compact vector representation, *not* the full float32 vector store which lives on disk).
_INDEX_MB_PER_GB = 29
#: Short timeout (s) for the doctor's reachability probe — fail fast, stay offline-friendly.
_PROBE_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Argument parser.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the ``aether-context`` console script.

    Exposes the ``init`` / ``doctor`` / ``bench`` subcommands plus a top-level ``--pool N``
    resize shorthand (so ``aether-context --pool 10`` works with no subcommand). Returns the
    parser; callers run ``parser.parse_args(argv)``.
    """
    parser = argparse.ArgumentParser(
        prog="aether-context",
        description=(
            "Unlimited Context — virtual memory for an LLM's attention. "
            "Init/resize the local pool, diagnose your setup, or run the bench."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aether-context {__version__}")
    # top-level resize shorthand: `aether-context --pool 10`
    parser.add_argument(
        "--pool", type=int, default=None, metavar="N",
        help="resize the pool to N GB (>=5), non-destructive; runs with no subcommand.",
    )
    parser.add_argument(
        "--dir", type=str, default=None, metavar="D",
        help="pool directory (default: ~/.aether-context).",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser(
        "init", help="initialize the pool (interactive slider on a tty; else --pool/env/5GB)."
    )
    p_init.add_argument("--pool", type=int, default=None, metavar="N", help="pool size in GB (>=5).")
    p_init.add_argument("--dir", type=str, default=None, metavar="D", help="pool directory.")

    p_doctor = subparsers.add_parser(
        "doctor", help="diagnose Ollama reachability, model pull, and RAM-vs-index (offline)."
    )
    p_doctor.add_argument("--model", type=str, default=None, metavar="M", help="model to check.")
    p_doctor.add_argument("--dir", type=str, default=None, metavar="D", help="pool directory.")
    p_doctor.add_argument("--host", type=str, default=None, metavar="URL", help="Ollama host.")

    p_bench = subparsers.add_parser(
        "bench", help="run the engine ON-vs-OFF bench (hermetic mock by default)."
    )
    p_bench.add_argument("--model", type=str, default="mock", metavar="M", help="model spec.")
    p_bench.add_argument("--quick", action="store_true", help="shorter build (CI smoke).")
    p_bench.add_argument("--json", action="store_true", help="machine-readable report.")

    return parser


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the matching command. Returns a process exit code.

    A bare invocation (no command, no ``--pool``) prints help and returns 0. Each command
    returns 0 on success and a non-zero code on a reported failure (so the script is usable in
    CI). Typed library errors are caught and rendered with their ``.hint``; nothing escapes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "bench":
            return _cmd_bench(args)
        # no subcommand: a top-level --pool is a resize; otherwise show help.
        if args.pool is not None:
            return _cmd_resize(args)
        parser.print_help()
        return 0
    except AetherContextError as exc:
        # typed, hinted failure: render it cleanly (never a traceback to the user).
        print(f"error: {exc.message}", file=sys.stderr)
        print(f"  fix: {exc.hint}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# init.
# ---------------------------------------------------------------------------
def _cmd_init(args: argparse.Namespace) -> int:
    """Initialize the pool config. Non-tty safe: prompt only on a real tty.

    Resolution order for the size: explicit ``--pool`` → interactive slider (tty only) →
    ``$AETHER_POOL_GB`` → the 5 GB default. A size below the floor is rejected with the reason.
    """
    pool_dir = _resolve_dir(args.dir)
    pool_gb = _resolve_pool_gb(getattr(args, "pool", None))
    cfg = _write_config(pool_dir, pool_gb)  # raises PoolBudgetError if < floor
    reach = reach_tokens(cfg.pool_gb)
    print(f"initialized pool at {cfg.dir}")
    print(f"  pool size: {cfg.pool_gb} GB  (reach ~= {reach / 1e9:.2f}B tokens)")
    print(f"  index: {cfg.index}   dim: {cfg.dim}   slice: {cfg.slice_tokens} tok")
    return 0


def _resolve_pool_gb(flag: int | None) -> int:
    """Resolve the pool size without ever blocking on a non-tty stdin.

    ``flag`` (``--pool``) wins. Else, only if stdin is an interactive tty do we run the slider.
    Else ``$AETHER_POOL_GB`` if set and numeric. Else the 5 GB default.
    """
    if flag is not None:
        return int(flag)
    if sys.stdin is not None and sys.stdin.isatty():
        return _prompt_pool_gb()
    env = os.environ.get(_ENV_POOL_GB)
    if env is not None and env.strip().isdigit():
        return int(env.strip())
    return POOL_GB_FLOOR


def _prompt_pool_gb() -> int:
    """Interactive pool-size selector (the README slider). Only called on a real tty.

    Shows the reach for a few sizes and reads one line. Empty input takes the 5 GB default;
    a value below the floor re-prompts once with the reason; EOF falls back to the default.
    """
    print("Choose a pool size (reach, not window). Bigger pool = more reach:")
    for gb in (5, 10, 15, 20):
        print(f"  {gb:>2} GB  ->  reach ~= {reach_tokens(gb) / 1e9:.2f}B tokens")
    for _attempt in range(2):
        try:
            raw = input(f"pool GB [default {POOL_GB_FLOOR}]: ").strip()
        except EOFError:
            return POOL_GB_FLOOR
        if not raw:
            return POOL_GB_FLOOR
        if raw.isdigit():
            value = int(raw)
            if value >= POOL_GB_FLOOR:
                return value
            print(f"  {value} GB is below the {POOL_GB_FLOOR} GB floor; pick at least {POOL_GB_FLOOR}.")
        else:
            print("  enter a whole number of GB (e.g. 5, 10, 20).")
    return POOL_GB_FLOOR


# ---------------------------------------------------------------------------
# resize (top-level --pool).
# ---------------------------------------------------------------------------
def _cmd_resize(args: argparse.Namespace) -> int:
    """Resize the pool to ``--pool N`` GB (non-destructive). Rejects ``N < floor``.

    "Non-destructive" = we only rewrite ``pool_gb`` in the persisted config; the on-disk
    vectors/sidecar payloads are left in place (a later pool open re-indexes around the new
    size). We load any existing config first so other settings (index/dim/...) are preserved.
    """
    pool_dir = _resolve_dir(args.dir)
    existing = PoolConfig.load(pool_dir)  # preserves index/dim/slice; defaults if absent
    cfg = _write_config(
        pool_dir, int(args.pool),
        index=existing.index, dim=existing.dim, slice_tokens=existing.slice_tokens,
        mode=existing.mode,
    )
    print(f"resized pool at {cfg.dir} to {cfg.pool_gb} GB "
          f"(reach ~= {reach_tokens(cfg.pool_gb) / 1e9:.2f}B tokens)")
    print("  re-index is non-destructive: your encoded slices are preserved.")
    return 0


def _write_config(pool_dir: Path, pool_gb: int, **fields: object) -> PoolConfig:
    """Build + persist a PoolConfig (validates the floor inside ``__post_init__``)."""
    cfg = PoolConfig(pool_gb=pool_gb, dir=pool_dir, **fields)  # type: ignore[arg-type]
    cfg.save()
    return cfg


def _resolve_dir(flag: str | None) -> Path:
    """Resolve the pool directory: ``--dir`` if given, else ``~/.aether-context``."""
    if flag:
        return Path(flag)
    return PoolConfig().dir


# ---------------------------------------------------------------------------
# doctor — runs fully offline, prints exact fixes.
# ---------------------------------------------------------------------------
def _cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the three common failure modes, printing the exact fix for each.

    Runs fully offline: the Ollama reachability probe has a short timeout and any network
    failure is rendered as a fixable condition (with ``ollama serve`` / ``ollama pull``), never
    raised. Returns 0 if everything checks out, 1 if any check found a problem.
    """
    host = _resolve_host(args.host)
    model = args.model
    pool_dir = _resolve_dir(args.dir)
    print("aether-context doctor")
    print(f"  ollama host: {host}")

    ok = True
    reachable = _probe_ollama(host)
    ok = _report_ollama(reachable, host) and ok
    ok = _report_model(reachable, host, model) and ok
    ok = _report_ram_vs_index(pool_dir) and ok

    print("")
    print("  all good." if ok else "  some checks need attention (see fixes above).")
    return 0 if ok else 1


def _resolve_host(flag: str | None) -> str:
    """Resolve the Ollama host: ``--host`` → ``$OLLAMA_HOST`` → default."""
    if flag:
        return flag.rstrip("/")
    env = os.environ.get(_ENV_OLLAMA_HOST)
    if env:
        return env.rstrip("/")
    return _DEFAULT_OLLAMA_HOST


def _probe_ollama(host: str) -> bool:
    """Best-effort reachability probe of the Ollama daemon. Never raises (offline-safe)."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _report_ollama(reachable: bool, host: str) -> bool:
    """Print the Ollama reachability check + its fix command. Returns True iff reachable."""
    if reachable:
        print(f"  [ok]   ollama daemon reachable at {host}")
        return True
    print(f"  [fail] ollama daemon not reachable at {host}")
    print("         fix: start it with `ollama serve`")
    return False


def _report_model(reachable: bool, host: str, model: str | None) -> bool:
    """Check whether ``model`` is pulled (only meaningful if the daemon is up).

    Always prints the exact ``ollama pull <model>`` fix command when a model was named, so the
    user sees the remedy even fully offline.
    """
    if model is None:
        print("  [skip] no --model given; pass --model qwen2.5 to check a specific model")
        return True
    if not reachable:
        print(f"  [fail] cannot check model '{model}' (daemon down)")
        print(f"         fix: start ollama then `ollama pull {model}`")
        return False
    pulled = _model_is_pulled(host, model)
    if pulled:
        print(f"  [ok]   model '{model}' is pulled")
        return True
    print(f"  [fail] model '{model}' is not pulled")
    print(f"         fix: `ollama pull {model}`  (or Session(model='ollama/{model}', pull=True))")
    return False


def _model_is_pulled(host: str, model: str) -> bool:
    """Return True iff ``model`` appears in ``/api/tags``. Offline-safe (False on any failure)."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    names = {m.get("name", "") for m in (body.get("models") or [])}
    # ollama tags carry a ':tag' suffix; match the bare name or any tag of it.
    base = model.split(":", 1)[0]
    return any(n == model or n.split(":", 1)[0] == base for n in names)


def _report_ram_vs_index(pool_dir: Path) -> bool:
    """Estimate the index RAM for the configured pool and compare to free system RAM.

    Reads the persisted ``PoolConfig`` (or defaults), computes the index RAM from the pool
    reach math (README table: ~145 MB at 5 GB), and warns if it would not comfortably fit in
    free RAM. Free RAM is probed best-effort; if it cannot be read we report the estimate only.
    """
    cfg = PoolConfig.load(pool_dir)
    index_bytes = _estimate_index_bytes(cfg)
    index_mb = index_bytes / (1024 * 1024)
    free_bytes = _free_ram_bytes()
    if free_bytes is None:
        print(f"  [ok]   index RAM estimate ~= {index_mb:.0f} MB "
              f"(free RAM unknown; could not probe)")
        return True
    free_mb = free_bytes / (1024 * 1024)
    # comfortable = index fits in well under half of free RAM.
    if index_bytes * 2 < free_bytes:
        print(f"  [ok]   index RAM ~= {index_mb:.0f} MB fits in {free_mb:.0f} MB free")
        return True
    print(f"  [warn] index RAM ~= {index_mb:.0f} MB vs only {free_mb:.0f} MB free")
    print("         fix: use a smaller --pool, or `--index tiered` to page the index")
    return False


def _estimate_index_bytes(cfg: PoolConfig) -> int:
    """In-RAM ANN index estimate for ``cfg``, matching the README table (~145 MB at 5 GB).

    The resident index scales with reach (more reach -> more slices -> a bigger graph), so the
    published table is linear in ``pool_gb`` at ~29 MB/GB. The full float32 vector store is far
    larger but lives on disk (mmap), so it is not what bounds RAM — the in-RAM index is.
    """
    return int(cfg.pool_gb * _INDEX_MB_PER_GB * 1024 * 1024)


def _free_ram_bytes() -> int | None:
    """Best-effort free-RAM probe using only the standard library. None if unavailable.

    Tries ``os.sysconf`` (POSIX) then a Windows ctypes call. Never raises — a probe failure
    simply yields ``None`` and the report degrades gracefully.
    """
    posix = _free_ram_posix()
    if posix is not None:
        return posix
    return _free_ram_windows()


def _free_ram_posix() -> int | None:
    """POSIX free-RAM via ``os.sysconf`` (available pages * page size). None off-POSIX."""
    try:
        names = getattr(os, "sysconf_names", {})
        sysconf = getattr(os, "sysconf", None)  # absent on Windows
        if sysconf is not None and "SC_AVPHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
            pages = sysconf("SC_AVPHYS_PAGES")
            page_size = sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages) * int(page_size)
    except (AttributeError, ValueError, OSError):
        return None
    return None


def _free_ram_windows() -> int | None:
    """Windows free-RAM via ``GlobalMemoryStatusEx`` (ctypes). None on non-Windows/failure."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys)
    except (OSError, AttributeError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# bench — delegate to bench/drift_vs_window.py.
# ---------------------------------------------------------------------------
def _cmd_bench(args: argparse.Namespace) -> int:
    """Delegate to ``bench/drift_vs_window.py``. Returns its exit code.

    The bench lives outside the importable package (it's a script under ``bench/``), so we load
    it by file path. If it cannot be located (e.g. an installed wheel that omits ``bench/``) we
    print an actionable hint rather than crashing.
    """
    module = _load_bench_module()
    if module is None:
        print("bench script not found (bench/drift_vs_window.py).", file=sys.stderr)
        print("  fix: run from a source checkout, or `python bench/drift_vs_window.py`.",
              file=sys.stderr)
        return 1
    bench_argv: list[str] = ["--model", str(args.model)]
    if args.quick:
        bench_argv.append("--quick")
    if getattr(args, "json", False):
        bench_argv.append("--json")
    return int(module.main(bench_argv))


def _load_bench_module() -> ModuleType | None:
    """Locate + import ``bench/drift_vs_window.py`` by file path. None if not found."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "bench" / "drift_vs_window.py",  # repo checkout: <root>/bench/
        Path.cwd() / "bench" / "drift_vs_window.py",          # invoked from repo root
    ]
    for path in candidates:
        if path.is_file():
            mod_name = "aether_context_bench"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Register before exec so the module's own dataclasses can resolve
            # ``cls.__module__`` via ``sys.modules`` during class creation.
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:  # noqa: BLE001 - a broken bench file shouldn't crash the CLI
                sys.modules.pop(mod_name, None)
                raise
            return module
    return None


__all__ = ["main", "build_parser"]
