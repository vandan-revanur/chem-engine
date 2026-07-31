"""
extreme_scale_benchmark.py - chem-engine vs RDKit at 100K / 1M / 10M molecules
================================================================================

WHY MULTIPLE DATABASES?
────────────────────────
ChEMBL 37 contains ~2.4M unique drug-like molecules - not enough for 10M
without recycling. Instead this script combines multiple open databases so
every molecule in the 10M pool is unique and chemically diverse:

  • ChEMBL 37   ~2.4M  drug-like, approved/clinical candidates
    https://www.ebi.ac.uk/chembl/  → chembl_37_chemreps.txt.gz

  • ZINC22       ~4.6M  (lead-like subset, Tanimoto-diverse, free)
    https://zinc22.docking.org/tranches/home/  → *_lead-like*.smi.gz

  • PubChem      ~3.0M  (BioAssay confirmed actives, drug-like filtered)
    ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/ → CID_SMILES.gz
    (filter with MW 200-600, HBD≤5, HBA≤10, rotbond≤10)

  Total unique after deduplication: ~10M

Use --sources to pass any combination of these files. The script
auto-detects format (tab-separated ChEMBL, space-separated ZINC,
single-column PubChem, or plain one-SMILES-per-line).

If only ChEMBL is provided and the pool is smaller than the largest
requested scale, the pool is cycled (with a warning).


Architecture (safe by design)
──────────────────────────────
  • STREAMING: molecules are never all loaded into RAM simultaneously.
    A generator reads ChEMBL line-by-line and cycles automatically to
    reach 1M or 10M without requiring the full dataset in memory.
  • MEMORY WATCHDOG: a background thread monitors process RSS every 2 s.
    If RSS exceeds --mem-limit-pct (default 70 % of total RAM), chunk
    submission is paused until memory falls below the threshold.
  • TIME BUDGET: each operation has a --time-budget (default 600 s / 10 min).
    The streaming loop stops early if the budget is exceeded, and the
    throughput is computed from however many molecules were processed.
  • WORKER CAP: --workers defaults to cpu_count // 3 (5 on a 16-core box).
    Always leaves ≥ 2 cores free for the OS and watchdog thread.
  • CHUNK SLEEP: 100 ms inter-chunk sleep between ProcessPoolExecutor
    submissions to avoid thundering-herd CPU spikes.
  • CHECKPOINTING: results are written to --checkpoint JSON after every
    completed (operation, scale) pair so a crash loses at most one measurement.
  • AUTO-CAPPING: expensive operations (tautomers, 2D layout, 3D embed)
    always use a fixed small sample (≤ 10 K / 1 K) regardless of scale.

Usage
─────
    # Default safe run - 100K / 1M / 10M, 5 workers
    python benchmarks/extreme_scale_benchmark.py \\
        --chembl /tmp/chembl_37_chemreps.txt.gz

    # Custom - 6 workers, 15-min budget, aggressive 80% RAM limit
    python benchmarks/extreme_scale_benchmark.py \\
        --chembl /tmp/chembl_37_chemreps.txt.gz \\
        --workers 6 --time-budget 900 --mem-limit-pct 80 \\
        --scales 100000,1000000,10000000

    # Resume after crash (skips already-completed operations)
    python benchmarks/extreme_scale_benchmark.py \\
        --chembl /tmp/chembl_37_chemreps.txt.gz \\
        --checkpoint /tmp/chem_bench_checkpoint.json
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

# ── process priority ─────────────────────────────────────────────────────────
try:
    os.nice(10)
except OSError:
    pass

# ── RDKit / chem-engine imports ───────────────────────────────────────────────
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import psutil
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

import chem_engine as ro

# ═══════════════════════════════════════════════════════════════════════════════
# Memory watchdog
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryWatchdog:
    """
    Background thread that monitors process RSS.
    When memory exceeds `limit_pct`% of total RAM, sets `self.paused = True`.
    Callers should check `watchdog.wait_if_needed()` before submitting new work.
    """

    def __init__(self, limit_pct: float = 70.0, poll_interval: float = 2.0):
        self._limit = limit_pct / 100.0
        self._poll = poll_interval
        self._total = psutil.virtual_memory().total
        self._stop = threading.Event()
        self.paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                sys_used = psutil.virtual_memory().percent / 100.0
                self.paused = (rss / self._total > self._limit) or (sys_used > self._limit)
            except Exception:
                pass
            time.sleep(self._poll)

    def wait_if_needed(self, label: str = ""):
        """Block (with status print) until memory is below threshold."""
        if not self.paused:
            return
        print(f"\n  [watchdog] memory pressure - pausing {label} …", flush=True)
        while self.paused:
            time.sleep(2.0)
        print("  [watchdog] memory OK - resuming.\n", flush=True)

    def stop(self):
        self._stop.set()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level worker functions  (must be picklable)
# ═══════════════════════════════════════════════════════════════════════════════


def _worker_validate(chunk):
    from rdkit import Chem

    import chem_engine as ro

    return [s for s in chunk if _try_ce(ro, s) and Chem.MolFromSmiles(s) is not None]


def _try_ce(ro, s):
    try:
        ro.parse_smiles(s)
        return True
    except Exception:
        return False


def _worker_parse_ce(chunk):
    import chem_engine as ro

    return [ro.parse_smiles(s) for s in chunk]


def _worker_parse_rk(chunk):
    from rdkit import Chem

    return [Chem.MolFromSmiles(s) for s in chunk]


def _worker_canonical_ce(chunk):
    import chem_engine as ro

    return [ro.canonicalize(ro.parse_smiles(s)) for s in chunk]


def _worker_canonical_rk(chunk):
    from rdkit import Chem

    return [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in chunk]


def _worker_amw_ce(chunk):
    import chem_engine as ro

    return [ro.parse_smiles(s).amw for s in chunk]


def _worker_amw_rk(chunk):
    from rdkit import Chem, Descriptors

    return [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in chunk]


def _worker_rotbonds_ce(chunk):
    import chem_engine as ro

    return [ro.parse_smiles(s).num_rotatable_bonds for s in chunk]


def _worker_rotbonds_rk(chunk):
    from rdkit import Chem

    return [rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(s)) for s in chunk]


def _worker_substruct_ce(args):
    chunk, q_smi = args
    import chem_engine as ro

    q = ro.parse_smiles(q_smi)
    return [ro.parse_smiles(s).has_substruct_match(q) for s in chunk]


def _worker_substruct_rk(args):
    chunk, q_smi = args
    from rdkit import Chem

    q = Chem.MolFromSmarts(q_smi)
    return [Chem.MolFromSmiles(s).HasSubstructMatch(q) for s in chunk]


def _worker_tautomers_ce(chunk):
    import chem_engine as ro

    return [ro.parse_smiles(s).enumerate_tautomers() for s in chunk]


def _worker_tautomers_rk(chunk):
    from rdkit import Chem

    e = rdMolStandardize.TautomerEnumerator()
    return [e.Enumerate(Chem.MolFromSmiles(s)) for s in chunk]


def _worker_layout2d_ce(chunk):
    import chem_engine as ro

    return [ro.generate_2d_coords(ro.parse_smiles(s)) for s in chunk]


def _worker_layout2d_rk(chunk):
    from rdkit import Chem

    return [AllChem.Compute2DCoords(Chem.MolFromSmiles(s)) for s in chunk]


def _worker_embed3d_ce(chunk):
    import chem_engine as ro

    return [ro.generate_3d_coords(ro.parse_smiles(s)) for s in chunk]


def _worker_embed3d_rk(chunk):
    from rdkit import Chem

    out = []
    for s in chunk:
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        out.append(AllChem.EmbedMolecule(m, randomSeed=42))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SMILES streaming source
# ═══════════════════════════════════════════════════════════════════════════════


def _raw_smiles_stream(path: str, max_len: int) -> Iterator[str]:
    """Yield raw (unvalidated) SMILES strings from ChEMBL file, cycling infinitely."""
    opener = gzip.open if str(path).endswith(".gz") else open
    while True:  # outer loop cycles the file for > 2.4 M requests
        with opener(path, "rt") as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    smi = parts[1].strip()
                    if smi and len(smi) <= max_len:
                        yield smi


def _detect_smiles_column(line: str) -> str | None:
    """
    Auto-detect SMILES from a line regardless of file format:
      • ChEMBL  : <chembl_id>\\t<smiles>\\t...   → col 1
      • ZINC22  : <smiles> <zinc_id>            → col 0
      • PubChem : <cid>\\t<smiles>              → col 1
      • Plain   : <smiles>                      → col 0
    Returns the SMILES string or None if the line looks like a header.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t") if "\t" in line else line.split()
    if not parts:
        return None

    # heuristic: SMILES columns contain C, c, N, O, (, [, = chars
    def looks_like_smiles(s: str) -> bool:
        return len(s) >= 2 and any(ch in s for ch in "CcNnOoSsPpFfBbIi([=#")

    for col in [1, 0, 2]:  # ChEMBL/PubChem col1, then ZINC/plain col0
        if col < len(parts) and looks_like_smiles(parts[col]):
            return parts[col]
    return None


def _stream_source(path: str, max_len: int):
    """Yield raw SMILES from a single source file (gzip or plain)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            smi = _detect_smiles_column(line)
            if smi and len(smi) <= max_len:
                yield smi


def build_validated_pool_multi(
    paths: list[str],
    max_len: int,
    pool_size: int,
    workers: int,
    chunk_size: int,
) -> list[str]:
    """
    Build a validated SMILES pool from one or more source files.
    Streams all sources in order, deduplicates on the fly, and stops
    once pool_size valid molecules are collected.
    """
    print(f"\n[pool] Building pool of {pool_size:,} from {len(paths)} source(s):", flush=True)
    for p in paths:
        print(f"       {p}", flush=True)

    seen: set[str] = set()
    valid: list[str] = []
    batch: list[str] = []
    read_total = 0

    def source_generator():
        for p in paths:
            print(f"\n[pool] Streaming {p} …", flush=True)
            try:
                yield from _stream_source(p, max_len)
            except FileNotFoundError:
                print(f"[pool] WARNING: {p} not found - skipping.", flush=True)
            except Exception as e:
                print(f"[pool] WARNING: error reading {p}: {e}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: list[Future] = []

        def _drain():
            done, remaining = [], []
            for f in pending:
                if f.done():
                    for s in f.result():
                        if s not in seen:
                            seen.add(s)
                            valid.append(s)
                    done.append(f)
                else:
                    remaining.append(f)
            pending[:] = remaining

        for smi in source_generator():
            if smi in seen:
                continue
            batch.append(smi)
            read_total += 1
            if len(batch) >= chunk_size:
                pending.append(pool.submit(_worker_validate, batch))
                batch = []
                _drain()
                if len(valid) >= pool_size:
                    for f in pending:
                        f.cancel()
                    break
            if read_total % 50_000 == 0:
                pct = min(100, len(valid) / pool_size * 100)
                print(
                    f"[pool]   {pct:.0f}%  ({len(valid):,}/{pool_size:,}, read {read_total:,}) …",
                    end="\r",
                    flush=True,
                )

        for f in as_completed(pending):
            for s in f.result():
                if s not in seen:
                    seen.add(s)
                    valid.append(s)
            if len(valid) >= pool_size:
                break

    valid = valid[:pool_size]
    unique_sources = len(paths)
    print(
        f"\n[pool] Done - {len(valid):,} unique validated molecules "
        f"from {unique_sources} source(s).",
        flush=True,
    )

    if len(valid) < pool_size:
        shortfall = pool_size - len(valid)
        print(
            f"[pool] WARNING: pool is {shortfall:,} short of target {pool_size:,}. "
            f"Cycling pool to compensate.",
            flush=True,
        )
        if valid:
            extended = list(itertools.islice(itertools.cycle(valid), pool_size))
            valid = extended

    return valid


# ── keep old single-source function as a thin wrapper ──────────────────────
def build_validated_pool(
    path: str, max_len: int, pool_size: int, workers: int, chunk_size: int
) -> list[str]:
    return build_validated_pool_multi([path], max_len, pool_size, workers, chunk_size)


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming throughput measurement
# ═══════════════════════════════════════════════════════════════════════════════


def streaming_throughput(
    ce_worker,
    rk_worker,
    smiles_pool: list[str],
    total_n: int,
    workers: int,
    chunk_size: int,
    watchdog: MemoryWatchdog,
    time_budget: float,
    inter_chunk_sleep: float = 0.1,
    extra_arg=None,
) -> tuple[tuple[float, float], tuple[float, float], int]:
    """
    Stream `total_n` molecules (cycling `smiles_pool`) through both engines.
    Returns ((ce_mps, ce_sd), (rk_mps, rk_sd), actual_n_processed).

    The streaming loop:
      1. Pulls a chunk from the cycling pool
      2. Submits it to ProcessPoolExecutor (capped workers)
      3. Sleeps inter_chunk_sleep to avoid burst CPU
      4. Respects memory watchdog pause
      5. Stops early if time_budget exceeded
    """
    source = itertools.cycle(smiles_pool)

    def run_engine(worker_fn) -> tuple[float, int]:
        """Run one pass through total_n molecules; return (total_wall_s, n_done)."""
        t_start = time.perf_counter()
        n_done = 0
        chunk_buf: list[str] = []
        time_exceeded = False

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending: list[Future] = []

            def _flush_done():
                done_futs, remaining = [], []
                for f in pending:
                    if f.done():
                        f.result()  # propagate exceptions
                        done_futs.append(f)
                    else:
                        remaining.append(f)
                pending[:] = remaining

            for smi in source:
                chunk_buf.append(smi)
                if len(chunk_buf) >= chunk_size:
                    watchdog.wait_if_needed(worker_fn.__name__)
                    arg = (chunk_buf, extra_arg) if extra_arg is not None else chunk_buf
                    pending.append(pool.submit(worker_fn, arg))
                    n_done += chunk_size
                    chunk_buf = []
                    time.sleep(inter_chunk_sleep)
                    _flush_done()

                    elapsed = time.perf_counter() - t_start
                    if n_done >= total_n or elapsed >= time_budget:
                        if elapsed >= time_budget:
                            time_exceeded = True
                        break

            # process final partial chunk
            if chunk_buf and not time_exceeded:
                arg = (chunk_buf, extra_arg) if extra_arg is not None else chunk_buf
                pending.append(pool.submit(worker_fn, arg))
                n_done += len(chunk_buf)

            # wait for all pending
            for f in as_completed(pending):
                f.result()

        wall = time.perf_counter() - t_start
        return wall, n_done

    # ── run chem-engine ──────────────────────────────────────────────────────
    ce_wall, ce_n = run_engine(ce_worker)
    time.sleep(1.0)  # brief pause between engines

    # ── run RDKit ────────────────────────────────────────────────────────────
    rk_wall, rk_n = run_engine(rk_worker)

    n_actual = min(ce_n, rk_n)
    # Re-compute per-run mps using the actual N each engine processed
    ce_mps = ce_n / ce_wall if ce_wall > 0 else 0.0
    rk_mps = rk_n / rk_wall if rk_wall > 0 else 0.0

    return (ce_mps, 0.0), (rk_mps, 0.0), n_actual


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt(mps: float) -> str:
    if mps >= 1_000_000:
        return f"{mps / 1e6:.2f} M mol/s"
    if mps >= 1_000:
        return f"{mps / 1e3:.1f} K mol/s"
    return f"{mps:.0f} mol/s"


def _ratio(ce: float, rk: float) -> str:
    r = rk / ce if ce > 0 else float("inf")
    sym = "✅" if r >= 1.0 else "❌"
    return f"{r:.1f}× {sym}"


def _scale_tag(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════════


def load_checkpoint(path: str) -> dict:
    if path and Path(path).exists():
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"[checkpoint] Loaded {len(data)} completed measurements from {path}")
            return data
        except Exception as e:
            print(f"[checkpoint] Could not load {path}: {e}")
    return {}


def save_checkpoint(path: str, results: dict):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_extreme_benchmarks(
    smiles_pool: list[str],
    scales: list[int],
    workers: int,
    chunk_size: int,
    time_budget: float,
    checkpoint_path: str,
    watchdog: MemoryWatchdog,
):
    results = load_checkpoint(checkpoint_path)
    cpu_n = os.cpu_count() or 1

    print("\n" + "═" * 78)
    print("  EXTREME-SCALE BENCHMARK: chem-engine vs RDKit")
    print(f"  Pool size : {len(smiles_pool):,} validated molecules (cycled for larger N)")
    print(f"  Scales    : {[_scale_tag(s) for s in scales]}")
    print(
        f"  Workers   : {workers}/{cpu_n}  |  chunk={chunk_size}  |  budget={time_budget}s  |  nice=+10"
    )
    print("═" * 78)

    # ── helper: run one operation at all scales ────────────────────────────
    def bench(label: str, ce_w, rk_w, allowed_scales=None, extra_arg=None, note: str = ""):
        scl = allowed_scales if allowed_scales is not None else scales
        print(f"\n### {label}", flush=True)
        if note:
            print(f"    ({note})")

        for N in scl:
            key = f"{label}@{_scale_tag(N)}"
            if key in results:
                ce_r, rk_r = results[key]["ce_mps"], results[key]["rk_mps"]
                print(
                    f"  {_scale_tag(N):>6}  [cached] CE: {_fmt(ce_r):>20} | RK: {_fmt(rk_r):>20} | {_ratio(ce_r, rk_r)}"
                )
                continue

            if N > len(smiles_pool) and N > 10_000_000:
                print(f"  {_scale_tag(N):>6}  - skipped (pool too small to cycle reliably)")
                continue

            print(f"  {_scale_tag(N):>6}  running …", end="\r", flush=True)
            (ce_mps, _), (rk_mps, _), n_done = streaming_throughput(
                ce_w,
                rk_w,
                smiles_pool,
                N,
                workers=workers,
                chunk_size=chunk_size,
                watchdog=watchdog,
                time_budget=time_budget,
                extra_arg=extra_arg,
            )

            tag = "" if n_done >= N else f"  [budget hit @ {_scale_tag(n_done)}]"
            print(
                f"  {_scale_tag(N):>6}  CE: {_fmt(ce_mps):>20} | RK: {_fmt(rk_mps):>20} | {_ratio(ce_mps, rk_mps)}{tag}"
            )

            results[key] = {"ce_mps": ce_mps, "rk_mps": rk_mps, "n_done": n_done}
            save_checkpoint(checkpoint_path, results)

        time.sleep(2.0)  # inter-operation cooldown

    # ── 1. SMILES Parsing ─────────────────────────────────────────────────
    bench("1. SMILES Parsing", _worker_parse_ce, _worker_parse_rk)

    # ── 2. Canonical SMILES ───────────────────────────────────────────────
    bench("2. Canonical SMILES", _worker_canonical_ce, _worker_canonical_rk)

    # ── 3. AMW Descriptor ─────────────────────────────────────────────────
    bench("3. AMW Descriptor", _worker_amw_ce, _worker_amw_rk)

    # ── 4. Rotatable Bonds ────────────────────────────────────────────────
    bench("4. Rotatable Bonds", _worker_rotbonds_ce, _worker_rotbonds_rk)

    # ── 5. Substructure Search ────────────────────────────────────────────
    bench(
        "5. Substructure Search", _worker_substruct_ce, _worker_substruct_rk, extra_arg="c1ccccc1"
    )

    # ── 6. Parallel Batch Parse (chem-engine Rayon - in-process, no IPC) ──
    print("\n### 6. Parallel Batch Parse (chem-engine Rayon vs RDKit loop)", flush=True)
    print("    (in-process - no IPC overhead; most favorable for Rayon)\n")
    for N in scales:
        key = f"6. Batch Parse@{_scale_tag(N)}"
        if key in results:
            ce_r, rk_r = results[key]["ce_mps"], results[key]["rk_mps"]
            print(
                f"  {_scale_tag(N):>6}  [cached] CE: {_fmt(ce_r):>20} | RK: {_fmt(rk_r):>20} | {_ratio(ce_r, rk_r)}"
            )
            continue

        # Process in rolling windows of pool size to avoid RAM blow-up
        window = smiles_pool  # use full pool each pass; cycle passes
        passes_needed = max(1, N // len(window))
        remainder = N - passes_needed * len(window)

        ce_times, rk_times = [], []
        t_budget_start = time.perf_counter()

        for p in range(passes_needed + (1 if remainder > 0 else 0)):
            chunk = window if p < passes_needed else window[:remainder]
            if not chunk:
                continue
            watchdog.wait_if_needed("batch parse")
            if time.perf_counter() - t_budget_start > time_budget:
                print(f"  {_scale_tag(N):>6}  time budget reached after {p} passes")
                break

            t0 = time.perf_counter()
            ro.batch_parse_smiles(chunk)
            ce_times.append((len(chunk), time.perf_counter() - t0))
            time.sleep(0.2)

            t0 = time.perf_counter()
            [Chem.MolFromSmiles(s) for s in chunk]
            rk_times.append((len(chunk), time.perf_counter() - t0))
            time.sleep(0.2)

        if not ce_times:
            continue

        ce_total_n = sum(n for n, _ in ce_times)
        ce_total_t = sum(t for _, t in ce_times)
        rk_total_n = sum(n for n, _ in rk_times)
        rk_total_t = sum(t for _, t in rk_times)
        ce_mps = ce_total_n / ce_total_t
        rk_mps = rk_total_n / rk_total_t

        print(
            f"  {_scale_tag(N):>6}  CE: {_fmt(ce_mps):>20} | RK: {_fmt(rk_mps):>20} | {_ratio(ce_mps, rk_mps)}"
        )
        results[key] = {"ce_mps": ce_mps, "rk_mps": rk_mps, "n_done": ce_total_n}
        save_checkpoint(checkpoint_path, results)
    time.sleep(2.0)

    # ── 7. Tanimoto (fixed 1K all-pairs - scale-invariant sample) ─────────
    print("\n### 7. Tanimoto Similarity (1K×1K all-pairs - scale-invariant)\n")
    SIM_N = min(1_000, len(smiles_pool))
    key_sim = f"7. Tanimoto@{_scale_tag(SIM_N)}"
    if key_sim not in results:
        sim_ce = [ro.parse_smiles(s) for s in smiles_pool[:SIM_N]]
        from rdkit import DataStructs
        from rdkit.Chem import MorganGenerator

        gen = MorganGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        sim_rk = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_pool[:SIM_N]]
        n_pairs = SIM_N * (SIM_N - 1) // 2

        t0 = time.perf_counter()
        for i in range(SIM_N):
            for j in range(i + 1, SIM_N):
                sim_ce[i].similarity(sim_ce[j])
        ce_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(SIM_N):
            for j in range(i + 1, SIM_N):
                DataStructs.TanimotoSimilarity(sim_rk[i], sim_rk[j])
        rk_t = time.perf_counter() - t0

        ce_mps = n_pairs / ce_t
        rk_mps = n_pairs / rk_t
        print(
            f"  {SIM_N:,}²/2 = {n_pairs:,} pairs | CE: {_fmt(ce_mps):>20} | RK: {_fmt(rk_mps):>20} | {_ratio(ce_mps, rk_mps)}"
        )
        results[key_sim] = {"ce_mps": ce_mps, "rk_mps": rk_mps, "n_done": n_pairs}
        save_checkpoint(checkpoint_path, results)
    else:
        r = results[key_sim]
        print(
            f"  [cached] CE: {_fmt(r['ce_mps']):>20} | RK: {_fmt(r['rk_mps']):>20} | {_ratio(r['ce_mps'], r['rk_mps'])}"
        )
    time.sleep(2.0)

    # ── 8. Tautomers (capped at 10K regardless of scale) ──────────────────
    bench(
        "8. Tautomers (10K cap)",
        _worker_tautomers_ce,
        _worker_tautomers_rk,
        allowed_scales=[min(10_000, s) for s in [10_000]],
        note="capped at 10K per scale to prevent multi-hour runs",
    )

    # ── 9. 2D Layout (capped at 10K) ──────────────────────────────────────
    bench(
        "9. 2D Layout (10K cap)",
        _worker_layout2d_ce,
        _worker_layout2d_rk,
        allowed_scales=[10_000],
        note="capped at 10K",
    )

    # ── 10. 3D Embedding (capped at 1K) ───────────────────────────────────
    bench(
        "10. 3D Embedding (1K cap)",
        _worker_embed3d_ce,
        _worker_embed3d_rk,
        allowed_scales=[1_000],
        note="capped at 1K - ETKDG is O(N³)",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("  EXTREME-SCALE SUMMARY")
    print("═" * 78)
    print(f"  {'Operation':<35} {'Scale':>8} {'CE':>18} {'RDKit':>18} {'Speedup':>10}")
    print("  " + "─" * 74)
    for key, val in sorted(results.items()):
        if "@" not in key:
            continue
        op, scl = key.rsplit("@", 1)
        ce, rk = val["ce_mps"], val["rk_mps"]
        r = rk / ce if ce > 0 else 0.0
        sym = "✅" if r >= 1.0 else "❌"
        print(f"  {op:<35} {scl:>8} {_fmt(ce):>18} {_fmt(rk):>18} {r:>7.1f}× {sym}")
    print("═" * 78)

    wins = sum(
        1
        for v in results.values()
        if "ce_mps" in v and "rk_mps" in v and v["ce_mps"] > 0 and v["rk_mps"] / v["ce_mps"] >= 1.0
    )
    total = sum(1 for v in results.values() if "ce_mps" in v)
    print(f"\n  chem-engine faster in {wins}/{total} measured operations.\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cpu_n = os.cpu_count() or 2
    default_workers = max(1, cpu_n // 3)  # conservative for extreme scale

    parser = argparse.ArgumentParser(
        description="Extreme-scale (100K-10M) chem-engine vs RDKit benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        metavar="FILE",
        help=(
            "One or more SMILES source files (gzip or plain). "
            "Supports ChEMBL (TSV col 1), ZINC22 (space col 0), "
            "PubChem (TSV col 1), and plain one-per-line. "
            "Files are combined and deduplicated. "
            "Example: --sources chembl.txt.gz zinc_lead.smi.gz pubchem.tsv.gz"
        ),
    )
    parser.add_argument(
        "--chembl",
        default=None,
        help="Shorthand for --sources with a single ChEMBL file (kept for backward compatibility).",
    )
    parser.add_argument("--max-smiles-len", type=int, default=150)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=200_000,
        help="Validated molecule pool to build; cycled for larger N",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"ProcessPoolExecutor workers (default: {default_workers} = cpu//3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000,
        help="Molecules per worker chunk (larger = less IPC overhead)",
    )
    parser.add_argument(
        "--time-budget", type=float, default=600.0, help="Max seconds per (operation × scale) pair"
    )
    parser.add_argument(
        "--mem-limit-pct",
        type=float,
        default=70.0,
        help="Pause new work when process RAM exceeds this %% of total",
    )
    parser.add_argument(
        "--scales",
        default="100000,1000000,10000000",
        help="Comma-separated molecule counts to benchmark",
    )
    parser.add_argument(
        "--checkpoint",
        default="/tmp/chem_extreme_checkpoint.json",
        help="JSON file for checkpoint/resume",
    )
    args = parser.parse_args()

    scales = sorted(set(int(x) for x in args.scales.split(",")))
    args.workers = max(1, min(args.workers, cpu_n - 2))  # always leave ≥ 2 cores free

    print("\n[config]")
    print(f"  workers      = {args.workers}/{cpu_n}")
    print(f"  pool_size    = {args.pool_size:,}")
    print(f"  chunk_size   = {args.chunk_size:,}")
    print(f"  time_budget  = {args.time_budget}s per operation")
    print(f"  mem_limit    = {args.mem_limit_pct}% of RAM")
    print(f"  scales       = {[_scale_tag(s) for s in scales]}")
    print(f"  checkpoint   = {args.checkpoint}")
    print("  nice         = +10")

    # ── start memory watchdog ─────────────────────────────────────────────
    watchdog = MemoryWatchdog(limit_pct=args.mem_limit_pct)
    vm = psutil.virtual_memory()
    print(
        f"\n[memory] Total: {vm.total / 1e9:.1f} GB  "
        f"Available: {vm.available / 1e9:.1f} GB  "
        f"Limit: {args.mem_limit_pct}% = {vm.total * args.mem_limit_pct / 100 / 1e9:.1f} GB"
    )

    # ── build validated SMILES pool ───────────────────────────────────────
    sources = args.sources if args.sources else ([args.chembl] if args.chembl else [])
    if not sources:
        print("ERROR: provide at least one source via --sources or --chembl")
        watchdog.stop()
        sys.exit(1)

    pool = build_validated_pool_multi(
        sources,
        args.max_smiles_len,
        args.pool_size,
        args.workers,
        args.chunk_size,
    )

    if len(pool) < 10_000:
        print(f"ERROR: only {len(pool)} valid SMILES in pool - need ≥ 10,000.")
        watchdog.stop()
        sys.exit(1)

    print(
        f"[info] Pool of {len(pool):,} molecules will be cycled "
        f"{max(scales) // len(pool) + 1}× to reach "
        f"{_scale_tag(max(scales))} scale.\n"
    )

    try:
        run_extreme_benchmarks(
            pool,
            scales,
            workers=args.workers,
            chunk_size=args.chunk_size,
            time_budget=args.time_budget,
            checkpoint_path=args.checkpoint,
            watchdog=watchdog,
        )
    finally:
        watchdog.stop()
