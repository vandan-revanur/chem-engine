# chem-engine

**Rust-native cheminformatics with Python bindings - a fast, memory-safe alternative to RDKit core functions.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

---

## Why chem-engine?

> "We envision this being done in a completely RDKit-compatible way so teams can benefit from
> the compute optimization without changing their workflows." - Ari Wagen, Computational Chemist

RDKit is the industry standard for cheminformatics but its Python layer introduces overhead that
compounds at scale. `chem-engine` reimplements core RDKit-style operations in Rust - memory-safe,
parallel, and 5-64x faster - while staying fully interoperable with existing RDKit workflows.

### Benchmark highlights (vs RDKit, drug-like molecules)

| Operation | Speedup |
|---|---|
| Batch SMILES parse (Rayon, 10M mols) | **64x** |
| 3D coordinate generation (ETKDG-like) | **37x** |
| Rotatable bond count | **35x** |
| Canonical SMILES | **9x** |
| SMILES parsing | **7-8x** |
| Substructure search | **2.5x** |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full throughput tables at 1K, 50K, 1M and 10M scales.

---

## Installation

### From source (requires Rust toolchain + maturin)

```bash
# Install Rust: https://rustup.rs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install chem-engine
git clone https://github.com/vandan-revanur/chem-engine.git
cd chem-engine
maturin develop --release
```

---

## Quick start

```python
import chem_engine as ce

# Parse SMILES
mol = ce.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
print(mol.num_atoms)              # 13
print(mol.num_bonds)              # 13
print(mol.amw)                    # ~180 (heavy atoms only)
print(mol.num_rotatable_bonds)    # 3

# Canonical SMILES
can = ce.canonicalize(mol)
print(can)   # deterministic string

# 2D layout
mol2d = ce.generate_2d_coords(mol)
print(mol2d.coords_2d)   # [[x, y], ...]

# 3D embedding
mol3d = ce.generate_3d_coords(mol)
print(mol3d.coords_3d)   # [[x, y, z], ...]

# Fingerprint and similarity
fp = mol.get_fingerprint()          # 2048-bit ECFP2
sim = mol.similarity(mol)           # 1.0
print(sim)

# Substructure search
benzene = ce.parse_smiles("c1ccccc1")
print(mol.has_substruct_match(benzene))   # True

# Tautomers
acetone = ce.parse_smiles("CC(=O)C")
tautomers = acetone.enumerate_tautomers()
canonical = acetone.get_canonical_tautomer()

# Parallel batch parsing (uses all CPU cores via Rayon)
molecules = ce.batch_parse_smiles(["CCO", "CCCC", "c1ccccc1"] * 10_000)
```

---

## RDKit interoperability

```python
from chem_engine.utils import to_rdkit, from_rdkit
from rdkit import Chem

# chem-engine -> RDKit
rust_mol = ce.parse_smiles("c1ccccc1")
rd_mol = to_rdkit(rust_mol)
print(Chem.MolToSmiles(rd_mol))   # RDKit canonical SMILES

# RDKit -> chem-engine
rd_mol = Chem.MolFromSmiles("CC(=O)O")
rust_mol = from_rdkit(rd_mol)
print(rust_mol.num_atoms)         # 4

# Typical pattern: fast Rust pipeline, RDKit for accuracy-critical steps
rust_mol = ce.generate_3d_coords(ce.parse_smiles(smiles))
rd_mol = to_rdkit(rust_mol)       # hand off to RDKit
```

---

## Supported operations

| Feature | API |
|---|---|
| SMILES parsing (with validation) | `parse_smiles(smiles)` |
| Canonical SMILES (idempotent) | `canonicalize(mol)` |
| 2D force-directed layout | `generate_2d_coords(mol)` |
| 3D distance-geometry embedding | `generate_3d_coords(mol)` |
| AMW descriptor | `mol.amw` |
| Rotatable bond count | `mol.num_rotatable_bonds` |
| ECFP2 fingerprint (2048-bit) | `mol.get_fingerprint()` |
| Tanimoto similarity | `mol.similarity(other)` |
| Substructure search (VF-style) | `mol.has_substruct_match(query)` |
| Tautomer enumeration | `mol.enumerate_tautomers()` |
| Canonical tautomer | `mol.get_canonical_tautomer()` |
| Parallel batch parse (Rayon) | `batch_parse_smiles(smiles_list)` |
| RDKit conversion (bidirectional) | `to_rdkit(mol)`, `from_rdkit(rd_mol)` |
| Manual molecule construction | `RustMolecule()`, `Atom()`, `Bond()` |

Full API details in [docs/FEATURES.md](docs/FEATURES.md).

---

## Running the benchmarks

```bash
# Safe large-scale benchmark (1K / 10K / 50K molecules)
python benchmarks/large_scale_benchmark.py \
    --chembl /tmp/chembl_37_chemreps.txt.gz

# Extreme-scale streaming benchmark (100K / 1M / 10M)
python benchmarks/extreme_scale_benchmark.py \
    --chembl /tmp/chembl_37_chemreps.txt.gz

# Multi-database run (ChEMBL + ZINC22 + PubChem = ~10M unique molecules)
python benchmarks/extreme_scale_benchmark.py \
    --sources /tmp/chembl_37_chemreps.txt.gz \
              /tmp/zinc22_lead-like.smi.gz \
              /tmp/pubchem_actives.tsv.gz
```

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full benchmark methodology and results.

---

## Running the tests

```bash
# Full test suite (481 tests, 0 failures)
python -m pytest tests/ -q

# With verbose output
python -m pytest tests/ -v
```

Test files:

| File | What it covers |
|---|---|
| `tests/test_engine.py` | Original FR coverage (all 11 functional requirements) |
| `tests/test_known_molecules.py` | Pharma reference molecules with validated properties |
| `tests/test_edge_cases.py` | Edge cases, error paths, charged atoms |
| `tests/test_invariants.py` | Mathematical invariants (Tanimoto symmetry, coordinate quality) |
| `tests/test_correctness_vs_rdkit.py` | Cross-validation against RDKit for 20 reference molecules |
| `tests/test_substructure_extended.py` | Substructure search - bond types, pharma fragments |
| `tests/test_tautomers_extended.py` | Tautomer enumeration and canonical tautomer selection |

---

## Project structure

```
chem-engine/
|- src/                         # Rust source
|  |- lib.rs                    # PyO3 module exports
|  |- atom.rs                   # Atom struct (70+ elements)
|  |- bond.rs                   # Bond and BondType
|  |- molecule.rs               # RustMolecule (graph, descriptors, fingerprint)
|  `- algorithms/
|     |- smiles.rs              # SMILES parser + canonical SMILES emitter
|     `- layout.rs              # 2D/3D coordinate generation
|- chem_engine/                 # Python package
|  |- __init__.py               # Public API re-exports
|  `- utils.py                  # to_rdkit / from_rdkit helpers
|- benchmarks/                  # Benchmark scripts
|  |- large_scale_benchmark.py  # Safe parallel (50K)
|  `- extreme_scale_benchmark.py # Streaming (100K-10M)
|- tests/                       # Pytest suite (481 tests)
|- docs/                        # Documentation
|  |- BENCHMARKS.md             # Throughput tables
|  `- FEATURES.md               # Full feature reference
|- Cargo.toml                   # Rust dependencies (pyo3, rayon)
|- pyproject.toml               # Python package config (maturin)
`- LICENSE                      # MIT
```

---

## Architecture

`chem-engine` is built with [PyO3](https://pyo3.rs/) (Rust-Python FFI) and
[Maturin](https://www.maturin.rs/) (build tool). The core data structure is:

```
RustMolecule
  `- Arc<MoleculeData>
       |- atoms: Vec<Atom>
       |- bonds: Vec<Bond>
       |- coords_2d: Option<Vec<[f64; 2]>>
       `- coords_3d: Option<Vec<[f64; 3]>>
```

`Arc<MoleculeData>` makes cloning zero-copy. `batch_parse_smiles` dispatches
work to Rayon's work-stealing thread pool with `Send + Sync` safety guarantees.

---

## Known limitations

- **Tanimoto similarity** is 1.8x slower than RDKit (`Vec<bool>` fingerprint storage; fix: pack into `[u64; 32]`)
- **Tautomer coverage** is limited to keto-enol and amide-imidic rules (vs RDKit's 36 rules)
- **Stereochemistry** is not stored or propagated
- **Implicit H in AMW** uses only explicitly written H atoms

See [docs/FEATURES.md#limitations](docs/FEATURES.md#13-limitations-and-known-gaps) for the full list.

---

## License

[MIT](LICENSE) - Copyright (c) 2026 Vandan Revanur

