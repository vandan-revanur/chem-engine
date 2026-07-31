"""
Large-scale benchmark: chem-engine vs RDKit
Using ChEMBL 37 (https://www.ebi.ac.uk/chembl/) - ~2.4 M drug-like molecules.

Usage:
    python benchmarks/large_scale_benchmark.py --chembl /tmp/chembl_37_chemreps.txt.gz

Operations benchmarked at multiple scales (1K / 10K / 100K / 1M):
  1. SMILES parsing
  2. Canonical SMILES generation
  3. AMW descriptor
  4. Rotatable bond count
  5. Tanimoto fingerprint similarity (pairwise on subset)
  6. Tautomer enumeration (subset, expensive)
  7. Substructure search (subset)
  8. 2D coordinate generation (subset)
  9. 3D coordinate generation (subset)
 10. Parallel batch parse (chem-engine only)
"""

import argparse
import gzip
import statistics
import sys
import time
import warnings

# Suppress RDKit kekulization warnings for large-scale runs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

import chem_engine as ro

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_chembl_smiles(
    path: str, max_mols: int = 2_000_000, max_smiles_len: int = 150
) -> list[str]:
    """
    Load canonical SMILES from the ChEMBL chemreps TSV (gzip or plain).
    Filters out peptides / macromolecules (SMILES > max_smiles_len chars)
    and entries that chem-engine or RDKit cannot parse.
    Returns list of SMILES strings.
    """
    print(f"\n[load] Reading ChEMBL SMILES from {path} …", flush=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    raw = []
    with opener(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue  # header
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            smi = parts[1].strip()
            if not smi or len(smi) > max_smiles_len:
                continue
            raw.append(smi)
            if len(raw) >= max_mols:
                break

    print(f"[load] Read {len(raw):,} SMILES (len ≤ {max_smiles_len}) …", flush=True)

    # Validate: keep only those parseable by BOTH engines
    valid = []
    step = max(1, len(raw) // 20)
    for idx, smi in enumerate(raw):
        if idx % step == 0:
            pct = idx / len(raw) * 100
            print(f"[load]   validation {pct:.0f}% …", end="\r", flush=True)
        try:
            ro.parse_smiles(smi)
        except Exception:
            continue
        if Chem.MolFromSmiles(smi) is None:
            continue
        valid.append(smi)

    print(f"\n[load] Validation complete: {len(valid):,} mutually valid SMILES.\n", flush=True)
    return valid


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def throughput(fn, n_mols: int, n_reps: int = 3) -> tuple[float, float]:
    """Run fn() n_reps times; return (mean_mols_per_sec, stdev)."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mps_vals = [n_mols / t for t in times]
    return statistics.mean(mps_vals), statistics.stdev(mps_vals) if len(mps_vals) > 1 else 0.0


def fmt(mps: float, sd: float) -> str:
    if mps >= 1_000_000:
        return f"{mps / 1e6:.2f} M mol/s  (±{sd / 1e6:.2f})"
    if mps >= 1_000:
        return f"{mps / 1e3:.1f} K mol/s  (±{sd / 1e3:.1f})"
    return f"{mps:.0f} mol/s  (±{sd:.0f})"


def ratio_str(ce: float, rk: float) -> str:
    r = rk / ce if ce > 0 else float("inf")
    tag = f"{r:.1f}×  ✅" if r >= 1.0 else f"{r:.2f}×  ❌ (slower)"
    return tag


def run_benchmarks(smiles_all: list[str]):
    results = {}

    print("=" * 72)
    print("  LARGE-SCALE BENCHMARK: chem-engine vs RDKit")
    print(f"  Dataset: ChEMBL 37   Total available: {len(smiles_all):,} molecules")
    print("=" * 72)

    SCALES = [1_000, 10_000, 100_000]
    # Add 1M tier only if we have the data
    if len(smiles_all) >= 1_000_000:
        SCALES.append(1_000_000)

    # -----------------------------------------------------------------------
    # 1. SMILES Parsing
    # -----------------------------------------------------------------------
    print("\n### 1. SMILES Parsing\n")
    parse_rows = []
    for N in SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(5, 20_000 // N))

        ce_mps, ce_sd = throughput(lambda s=subset: [ro.parse_smiles(x) for x in s], N, n_reps)
        rk_mps, rk_sd = throughput(lambda s=subset: [Chem.MolFromSmiles(x) for x in s], N, n_reps)

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        parse_rows.append((N, ce_mps, rk_mps))
        results[f"parse_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 2. Canonical SMILES
    # -----------------------------------------------------------------------
    print("\n### 2. Canonical SMILES Generation\n")
    for N in SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(3, 10_000 // N))
        ce_mols = [ro.parse_smiles(s) for s in subset]
        rd_mols = [Chem.MolFromSmiles(s) for s in subset]

        ce_mps, ce_sd = throughput(lambda m=ce_mols: [ro.canonicalize(x) for x in m], N, n_reps)
        rk_mps, rk_sd = throughput(lambda m=rd_mols: [Chem.MolToSmiles(x) for x in m], N, n_reps)

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"canonical_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 3. AMW Descriptor
    # -----------------------------------------------------------------------
    print("\n### 3. Average Molecular Weight (AMW)\n")
    for N in SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(5, 50_000 // N))
        ce_mols = [ro.parse_smiles(s) for s in subset]
        rd_mols = [Chem.MolFromSmiles(s) for s in subset]

        ce_mps, ce_sd = throughput(lambda m=ce_mols: [x.amw for x in m], N, n_reps)
        rk_mps, rk_sd = throughput(lambda m=rd_mols: [Descriptors.MolWt(x) for x in m], N, n_reps)

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"amw_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 4. Rotatable Bonds
    # -----------------------------------------------------------------------
    print("\n### 4. Rotatable Bond Count\n")
    for N in SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(5, 20_000 // N))
        ce_mols = [ro.parse_smiles(s) for s in subset]
        rd_mols = [Chem.MolFromSmiles(s) for s in subset]

        ce_mps, ce_sd = throughput(lambda m=ce_mols: [x.num_rotatable_bonds for x in m], N, n_reps)
        rk_mps, rk_sd = throughput(
            lambda m=rd_mols: [rdMolDescriptors.CalcNumRotatableBonds(x) for x in m], N, n_reps
        )

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"rotbonds_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 5. Tanimoto Similarity (pairwise on 1K subset)
    # -----------------------------------------------------------------------
    print("\n### 5. Tanimoto Fingerprint Similarity (1K × 1K all-pairs)\n")
    SIM_N = min(1_000, len(smiles_all))
    sim_subset = smiles_all[:SIM_N]
    ce_mols_sim = [ro.parse_smiles(s) for s in sim_subset]
    from rdkit.Chem import MorganGenerator

    gen = MorganGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rd_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in sim_subset]
    from rdkit import DataStructs

    def ce_sim_fn():
        for i in range(len(ce_mols_sim)):
            for j in range(i + 1, len(ce_mols_sim)):
                ce_mols_sim[i].similarity(ce_mols_sim[j])

    def rk_sim_fn():
        for i in range(len(rd_fps)):
            for j in range(i + 1, len(rd_fps)):
                DataStructs.TanimotoSimilarity(rd_fps[i], rd_fps[j])

    n_pairs = SIM_N * (SIM_N - 1) // 2
    ce_mps, ce_sd = throughput(ce_sim_fn, n_pairs, 3)
    rk_mps, rk_sd = throughput(rk_sim_fn, n_pairs, 3)
    spd = ratio_str(ce_mps, rk_mps)
    print(
        f"  N={SIM_N:>8,}²/2 = {n_pairs:,} pairs | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
    )
    results[f"tanimoto_{SIM_N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 6. Tautomer Enumeration (10K subset - expensive for RDKit)
    # -----------------------------------------------------------------------
    print("\n### 6. Tautomer Enumeration (up to 10K molecules)\n")
    TAUT_SCALES = [1_000, 10_000]
    enumerator = rdMolStandardize.TautomerEnumerator()
    for N in TAUT_SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(3, 3_000 // N))
        ce_mols_t = [ro.parse_smiles(s) for s in subset]
        rd_mols_t = [Chem.MolFromSmiles(s) for s in subset]

        ce_mps, ce_sd = throughput(
            lambda m=ce_mols_t: [x.enumerate_tautomers() for x in m], N, n_reps
        )
        rk_mps, rk_sd = throughput(
            lambda m=rd_mols_t, e=enumerator: [e.Enumerate(x) for x in m], N, n_reps
        )

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"tautomers_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 7. Substructure Search (benzene ring query vs 10K targets)
    # -----------------------------------------------------------------------
    print("\n### 7. Substructure Search (benzene ring query)\n")
    SS_SCALES = [1_000, 10_000, 100_000]
    query_ce = ro.parse_smiles("c1ccccc1")
    query_rk = Chem.MolFromSmarts("c1ccccc1")
    for N in SS_SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(5, 10_000 // N))
        ce_targets = [ro.parse_smiles(s) for s in subset]
        rd_targets = [Chem.MolFromSmiles(s) for s in subset]

        ce_mps, ce_sd = throughput(
            lambda t=ce_targets, q=query_ce: [x.has_substruct_match(q) for x in t], N, n_reps
        )
        rk_mps, rk_sd = throughput(
            lambda t=rd_targets, q=query_rk: [x.HasSubstructMatch(q) for x in t], N, n_reps
        )

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"substruct_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 8. 2D Coordinate Generation (1K subset)
    # -----------------------------------------------------------------------
    print("\n### 8. 2D Coordinate Generation (up to 10K molecules)\n")
    LAYOUT_SCALES = [1_000, 10_000]
    for N in LAYOUT_SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(3, 3_000 // N))
        ce_mols_2d = [ro.parse_smiles(s) for s in subset]
        rd_mols_2d = [Chem.MolFromSmiles(s) for s in subset]

        ce_mps, ce_sd = throughput(
            lambda m=ce_mols_2d: [ro.generate_2d_coords(x) for x in m], N, n_reps
        )
        rk_mps, rk_sd = throughput(
            lambda m=rd_mols_2d: [AllChem.Compute2DCoords(x) for x in m], N, n_reps
        )

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"layout2d_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 9. 3D Embedding (1K subset - ETKDG is slow in RDKit)
    # -----------------------------------------------------------------------
    print("\n### 9. 3D Coordinate Generation (ETKDG-like, up to 1K molecules)\n")
    EMBED_SCALES = [500, 1_000]
    for N in EMBED_SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(3, 1_500 // N))
        ce_mols_3d = [ro.parse_smiles(s) for s in subset]

        def rk_3d_fn(smis=subset):
            for s in smis:
                m = Chem.AddHs(Chem.MolFromSmiles(s))
                AllChem.EmbedMolecule(m, randomSeed=42)

        ce_mps, ce_sd = throughput(
            lambda m=ce_mols_3d: [ro.generate_3d_coords(x) for x in m], N, n_reps
        )
        rk_mps, rk_sd = throughput(rk_3d_fn, N, n_reps)

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"embed3d_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # 10. Parallel Batch Parse (chem-engine Rayon vs RDKit Python loop)
    # -----------------------------------------------------------------------
    print("\n### 10. Parallel Batch Parse (chem-engine Rayon vs RDKit Python loop)\n")
    for N in SCALES:
        if N > len(smiles_all):
            continue
        subset = smiles_all[:N]
        n_reps = max(1, min(5, 20_000 // N))

        ce_mps, ce_sd = throughput(lambda s=subset: ro.batch_parse_smiles(s), N, n_reps)
        rk_mps, rk_sd = throughput(lambda s=subset: [Chem.MolFromSmiles(x) for x in s], N, n_reps)

        spd = ratio_str(ce_mps, rk_mps)
        print(
            f"  N={N:>8,} | CE: {fmt(ce_mps, ce_sd):>30} | RDKit: {fmt(rk_mps, rk_sd):>30} | Speedup: {spd}"
        )
        results[f"batch_{N}"] = (ce_mps, rk_mps)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY - throughput speedup ratios (chem-engine / RDKit)")
    print("=" * 72)
    print(f"{'Operation':<35} {'Scale':>10} {'CE (mol/s)':>15} {'RK (mol/s)':>15} {'Speedup':>12}")
    print("-" * 72)
    for key, (ce, rk) in results.items():
        op, n = key.rsplit("_", 1)
        r = rk / ce if ce > 0 else 0
        sym = "✅" if r >= 1.0 else "❌"
        print(f"  {op:<33} {int(n):>10,} {ce:>15,.0f} {rk:>15,.0f} {r:>9.1f}× {sym}")
    print("=" * 72)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chembl",
        default="/tmp/chembl_37_chemreps.txt.gz",
        help="Path to chembl_37_chemreps.txt[.gz]",
    )
    parser.add_argument("--max-mols", type=int, default=1_200_000, help="Maximum molecules to load")
    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=150,
        help="Filter out SMILES longer than this (removes macromolecules)",
    )
    args = parser.parse_args()

    smiles = load_chembl_smiles(args.chembl, args.max_mols, args.max_smiles_len)
    if len(smiles) < 1_000:
        print(f"ERROR: only {len(smiles)} valid SMILES loaded - need at least 1,000.")
        sys.exit(1)

    run_benchmarks(smiles)
