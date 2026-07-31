# chem-engine: Supported Features

> Version 0.1.0 - Rust-native cheminformatics with Python bindings via PyO3

---

## Table of Contents

1. [SMILES Parsing](#1-smiles-parsing)
2. [Molecular Graph API](#2-molecular-graph-api)
3. [Canonical SMILES](#3-canonical-smiles)
4. [2D Coordinate Generation](#4-2d-coordinate-generation)
5. [3D Coordinate Generation](#5-3d-coordinate-generation)
6. [Molecular Descriptors](#6-molecular-descriptors)
7. [Fingerprints and Similarity](#7-fingerprints-and-similarity)
8. [Substructure Search](#8-substructure-search)
9. [Tautomer Enumeration](#9-tautomer-enumeration)
10. [Parallel Batch Processing](#10-parallel-batch-processing)
11. [RDKit Interoperability](#11-rdkit-interoperability)
12. [Supported Elements](#12-supported-elements)
13. [Limitations and Known Gaps](#13-limitations-and-known-gaps)

---

## 1. SMILES Parsing

**Function:** `chem_engine.parse_smiles(smiles: str) -> RustMolecule`

Parses a SMILES string into a `RustMolecule` graph. Uses a hand-written recursive-descent parser
that is ~5-8x faster than RDKit's parser for typical drug-like molecules.

### Supported SMILES features

| Feature | Example | Supported |
|---|---|---|
| Linear chains | `CCCC` | Yes |
| Branches | `CC(C)C` | Yes |
| Ring closures (single digit) | `C1CCCCC1` | Yes |
| Ring closures (two-digit `%NN`) | `C%10CCCCC%10` | Yes |
| Double bonds | `C=C`, `C=O` | Yes |
| Triple bonds | `C#C`, `C#N` | Yes |
| Aromatic atoms (lowercase) | `c1ccccc1` | Yes |
| Aromatic bonds | `c:1:c:c:c:c:c:1` | Yes |
| Bracket atoms | `[NH4+]`, `[O-]`, `[13C]` | Yes |
| Explicit hydrogens | `[H]O[H]` | Yes |
| Formal charges | `[Na+]`, `[Ca2+]`, `[O-]` | Yes |
| Isotope labels | `[13C]`, `[2H]` | Parsed, ignored |
| Stereo bonds (`/`, `\\`) | `C/C=C/C` | Parsed as single bond |
| Stereo centers (`@`, `@@`) | `[C@@H]` | Parsed, 3D not set |
| Disconnected fragments (`.`) | `[Na+].[Cl-]` | Yes |
| Wildcard atoms | `[*]` | Yes |

### Validation errors raised

| Invalid input | Error |
|---|---|
| Unknown organic-subset element | `SMILES parse error: unknown element 'X'` |
| Unknown bracket element | `SMILES parse error: unknown element '[Xx]'` |
| Unclosed ring | `SMILES parse error: unclosed ring closure(s): [1]` |
| Unclosed branch | `SMILES parse error: 1 unclosed branch(es) '('` |
| Unclosed bracket | `SMILES parse error: unclosed '[' bracket` |
| Empty / whitespace string | `SMILES parse error: empty string` |

### Quick example

```python
import chem_engine as ce

mol = ce.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
print(mol.num_atoms)   # 13
print(mol.num_bonds)   # 13
```

---

## 2. Molecular Graph API

**Class:** `chem_engine.RustMolecule`

The central data structure. Atoms and bonds are stored in an immutable
`Arc<MoleculeData>` so clones are zero-copy.

### RustMolecule properties

| Property | Type | Description |
|---|---|---|
| `num_atoms` | `int` | Number of heavy atoms |
| `num_bonds` | `int` | Number of bonds |
| `amw` | `float` | Average molecular weight (heavy atoms + explicit H only) |
| `num_rotatable_bonds` | `int` | Non-terminal, non-ring single bonds |
| `coords_2d` | `list[list[float]] or None` | Per-atom (x, y) coordinates |
| `coords_3d` | `list[list[float]] or None` | Per-atom (x, y, z) coordinates |

### RustMolecule methods

| Method | Returns | Description |
|---|---|---|
| `get_atom(idx)` | `Atom or None` | Atom at index `idx` |
| `get_bond(idx)` | `Bond or None` | Bond at index `idx` |
| `find_bond(u, v)` | `Bond or None` | Bond between atoms `u` and `v` |
| `add_atom(atom)` | - | Append an `Atom` |
| `add_bond(u, v, bond_type)` | - | Add a bond |
| `get_fingerprint()` | `list[bool]` | 2048-bit ECFP2 fingerprint |
| `similarity(other)` | `float` | Tanimoto coefficient (0.0-1.0) |
| `has_substruct_match(query)` | `bool` | Subgraph isomorphism check |
| `enumerate_tautomers()` | `list[RustMolecule]` | All tautomers |
| `get_canonical_tautomer()` | `RustMolecule` | Highest-scoring tautomer |

**Coordinates are also settable:**
```python
mol.coords_2d = [[0.0, 0.0], [1.5, 0.0], ...]
mol.coords_3d = [[0.0, 0.0, 0.0], ...]
```

### Atom properties

**Class:** `chem_engine.Atom`

| Property | Type | Description |
|---|---|---|
| `atomic_number` | `int` | Proton count (e.g. 6 for C) |
| `symbol` | `str` | Element symbol (e.g. `"C"`, `"Na"`) |
| `formal_charge` | `int` | Formal charge (e.g. +1, -1) |
| `num_explicit_hs` | `int` | Explicit H count from bracket notation |
| `is_aromatic` | `bool` | True for aromatic atoms (lowercase SMILES) |
| `implicit_valence` | `int` | Valence used for implicit H inference |

**Constructor:** `Atom(atomic_number, formal_charge=0, num_explicit_hs=0, is_aromatic=False)`

### Bond properties

**Class:** `chem_engine.Bond`

| Property | Type | Description |
|---|---|---|
| `source_idx` | `int` | Index of first atom |
| `target_idx` | `int` | Index of second atom |
| `bond_type` | `BondType` | One of Single / Double / Triple / Aromatic |

**Enum:** `chem_engine.BondType`

| Value | Meaning |
|---|---|
| `BondType.Single` | Single bond (sp3) |
| `BondType.Double` | Double bond |
| `BondType.Triple` | Triple bond |
| `BondType.Aromatic` | Aromatic bond (between aromatic atoms) |

### Building molecules manually

```python
import chem_engine as ce

mol = ce.RustMolecule()
mol.add_atom(ce.Atom(6))             # carbon
mol.add_atom(ce.Atom(8))             # oxygen
mol.add_bond(0, 1, ce.BondType.Double)   # C=O (formaldehyde)
print(mol.num_atoms, mol.num_bonds)  # 2, 1
```

---

## 3. Canonical SMILES

**Function:** `chem_engine.canonicalize(mol: RustMolecule) -> str`

Generates a canonical SMILES string using a Morgan-rank DFS traversal.

- **Deterministic:** same structure always produces the same string
- **Idempotent:** `canonicalize(parse(canonicalize(parse(smi)))) == canonicalize(parse(smi))`
- **Ring numbers:** uses `%NN` format for ring closures >= 10
- **Note:** The canonical form is *internally consistent* but not guaranteed to match
  RDKit's InChI-derived canonical SMILES. Use `to_rdkit()` + `Chem.MolToSmiles()` when
  cross-tool equivalence is required.

```python
mol1 = ce.parse_smiles("CCO")
mol2 = ce.parse_smiles("OCC")
assert ce.canonicalize(mol1) == ce.canonicalize(mol2)   # True
```

**Speedup vs RDKit:** ~9x

---

## 4. 2D Coordinate Generation

**Function:** `chem_engine.generate_2d_coords(mol: RustMolecule) -> RustMolecule`

Assigns 2D (x, y) coordinates to each heavy atom using a **force-directed layout**:
- Circular initialization
- 100 iterations of spring + repulsion forces
- Returns a new `RustMolecule` with `coords_2d` populated

```python
mol = ce.parse_smiles("c1ccccc1")
mol = ce.generate_2d_coords(mol)
coords = mol.coords_2d           # list of [x, y] for each atom
```

**Speedup vs RDKit `Compute2DCoords`:** ~2.8x

---

## 5. 3D Coordinate Generation

**Function:** `chem_engine.generate_3d_coords(mol: RustMolecule) -> RustMolecule`

Assigns 3D (x, y, z) coordinates using a distance-geometry pipeline:

1. Floyd-Warshall shortest-path distances -> distance bounds matrix
2. Golden-angle spiral sphere initialisation (MDS-like)
3. 150-iteration distance-restraint force-field minimisation

Returns a new `RustMolecule` with `coords_3d` populated.

```python
mol = ce.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
mol = ce.generate_3d_coords(mol)
coords = mol.coords_3d    # list of [x, y, z] for each heavy atom
```

**Speedup vs RDKit `EmbedMolecule` (ETKDG):** ~37x  
**Note:** Operates on heavy atoms only (no explicit H). For full stereochemical
accuracy, use RDKit's ETKDG after converting via `to_rdkit()`.

---

## 6. Molecular Descriptors

Descriptors are computed as properties on `RustMolecule`.

### Average Molecular Weight (AMW)

```python
mol = ce.parse_smiles("[H]O[H]")
print(mol.amw)   # 18.015
```

AMW = sum of heavy-atom masses + explicit H masses (1.008 Da per H).  
**Important:** Implicit hydrogens on organic-subset atoms (e.g., `C` in `CCO`)
are not counted unless written explicitly as bracket atoms (`[CH4]`).
Use `[H]O[H]` form or convert via RDKit for full-precision MW.

**Speedup vs `Descriptors.MolWt`:** ~9x

### Rotatable Bond Count

```python
mol = ce.parse_smiles("CCCC")
print(mol.num_rotatable_bonds)   # 1
```

Definition: a single bond that is:
- Not in a ring (bridge detection via BFS)
- Not terminal (both atoms have degree > 1)
- Not aromatic / double / triple

**Speedup vs `CalcNumRotatableBonds`:** ~35x

---

## 7. Fingerprints and Similarity

### Fingerprint Generation

**Method:** `mol.get_fingerprint() -> list[bool]`

Produces a **2048-bit ECFP2-style Morgan fingerprint**:
- Round 0: per-atom hash of (atomic_number, degree, charge, explicit_H, is_aromatic)
- Round 1 + 2: FNV-1a propagation through sorted (bond_type, neighbour_hash) pairs
- Bits set at every round for every atom

```python
fp = mol.get_fingerprint()   # list of 2048 booleans
```

### Tanimoto Similarity

**Method:** `mol.similarity(other: RustMolecule) -> float`

Computes the Tanimoto (Jaccard) coefficient over the 2048-bit fingerprints.

```python
ethanol   = ce.parse_smiles("CCO")
ethylamine = ce.parse_smiles("CCN")
print(ethanol.similarity(ethylamine))   # ~0.20
print(ethanol.similarity(ethanol))      # 1.0
```

Properties:
- Range: [0.0, 1.0]
- Symmetric: `a.similarity(b) == b.similarity(a)`
- Self-similarity: always 1.0
- Empty molecule: always 0.0

**Known gap:** Tanimoto is ~1.8x slower than RDKit because the fingerprint
is stored as `Vec<bool>`. Upgrading to `[u64; 32]` packed bits would enable
SIMD popcount and close this gap (5-10x improvement expected).

---

## 8. Substructure Search

**Method:** `target.has_substruct_match(query: RustMolecule) -> bool`

Checks whether `query` is a subgraph of `target` using backtracking VF-style
atom-by-atom assignment with exact bond-type and aromaticity matching.

```python
benzene  = ce.parse_smiles("c1ccccc1")
aspirin  = ce.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
print(aspirin.has_substruct_match(benzene))   # True
print(benzene.has_substruct_match(aspirin))   # False
```

Matching rules:
- Atom type must match exactly (`atomic_number` + `is_aromatic`)
- Bond type must match exactly (Single / Double / Triple / Aromatic)
- `[C]` (aliphatic) does NOT match aromatic `c` atoms
- Empty query always returns `True`

**Speedup vs RDKit `HasSubstructMatch`:** ~2.5x

---

## 9. Tautomer Enumeration

### Enumerate all tautomers

**Method:** `mol.enumerate_tautomers() -> list[RustMolecule]`

Applies 1,3 proton-shift rules to generate tautomers:
- Keto-enol: `C=O` adjacent C-H -> `C-OH` + `C=C`
- Amide-imidic: `C(=O)N` adjacent N-H -> `C(-O)=N`

```python
acetone = ce.parse_smiles("CC(=O)C")
tautomers = acetone.enumerate_tautomers()
print(len(tautomers))   # >= 2 (keto + enol)
```

All tautomers have the same atom count as the input.

### Canonical tautomer

**Method:** `mol.get_canonical_tautomer() -> RustMolecule`

Returns the highest-scoring tautomer using a scoring function that prefers:
- Keto form (C=O) over enol (+15 points per carbonyl)
- Amide form (C=N) over imidic acid (+5 points)
- Aromatic atoms (+10 points each)

```python
canonical = mol.get_canonical_tautomer()
```

**Note:** Coverage is limited compared to RDKit's 36-rule SIE algorithm.
Heterocyclic systems (purines, pyrimidines, thiol-thione) may not be fully
enumerated. Use `to_rdkit()` + `TautomerEnumerator` for exhaustive coverage.

---

## 10. Parallel Batch Processing

**Function:** `chem_engine.batch_parse_smiles(smiles: list[str]) -> list[RustMolecule]`

Parses a list of SMILES strings in parallel using the **Rayon** work-stealing
thread pool (all available CPU cores).

```python
smiles_list = ["CCO", "CCCC", "c1ccccc1"] * 10_000
molecules = ce.batch_parse_smiles(smiles_list)   # parallel, uses all cores
```

Properties:
- Order is preserved: `molecules[i]` corresponds to `smiles_list[i]`
- Thread-safe (Rust `Send + Sync` on `RustMolecule`)
- Empty list returns empty list

**Speedup vs Python loop + RDKit:** ~64x at 10M scale (including Rayon parallelism)

---

## 11. RDKit Interoperability

**Module:** `chem_engine.utils`

Two-way conversion between `RustMolecule` and `rdkit.Chem.Mol` objects.
Uses V2000 MolBlock as the interchange format.

### RustMolecule -> RDKit

```python
from chem_engine.utils import to_rdkit

rust_mol = ce.parse_smiles("c1ccccc1")
rd_mol = to_rdkit(rust_mol)        # rdkit.Chem.Mol
smiles = Chem.MolToSmiles(rd_mol)  # RDKit canonical SMILES
```

### RDKit -> RustMolecule

```python
from chem_engine.utils import from_rdkit

rd_mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
rust_mol = from_rdkit(rd_mol)      # RustMolecule
```

### What is preserved in round-trip

| Property | Preserved |
|---|---|
| Atom types (element, charge) | Yes |
| Bond orders (Single/Double/Triple/Aromatic) | Yes |
| 2D coordinates | Yes |
| 3D coordinates | Yes |
| Atom count | Yes |
| Bond count | Yes |
| Stereochemistry | No (not stored in RustMolecule) |
| Atom map numbers | No |

### Typical pattern: compute in Rust, finalize in RDKit

```python
import chem_engine as ce
from chem_engine.utils import to_rdkit
from rdkit.Chem import AllChem, Descriptors

# Fast parse and 3D generation in Rust
rust_mol = ce.generate_3d_coords(ce.parse_smiles(smiles))

# Transfer to RDKit for full-accuracy downstream
rd_mol = to_rdkit(rust_mol)
mw = Descriptors.ExactMolWt(rd_mol)
```

---

## 12. Supported Elements

The parser recognises **all elements from H (1) to Bi (83)** in bracket notation.
The organic subset (without brackets) supports: B, C, N, O, F, P, S, Cl, Br, I.

### Organic subset (no brackets needed)

`B` `C` `N` `O` `F` `P` `S` `Cl` `Br` `I` `H`  
Aromatic variants: `b` `c` `n` `o` `p` `s` (and `as`, `se`)

### Full element table (bracket notation `[Na+]`, `[Fe]`, etc.)

| Period | Elements |
|---|---|
| 1 | H, He |
| 2 | Li, Be, B, C, N, O, F, Ne |
| 3 | Na, Mg, Al, Si, P, S, Cl, Ar |
| 4 | K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr |
| 5 | Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe |
| 6 | Cs, Ba, La, ..., Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi |

---

## 13. Limitations and Known Gaps

| Area | Limitation | Planned fix |
|---|---|---|
| **Tanimoto similarity** | 1.8x slower than RDKit; `Vec<bool>` storage prevents SIMD | Change to `[u64; 32]` packed bits |
| **Tautomer coverage** | Only keto-enol and amide-imidic 1,3 shifts; misses purines, pyrimidines, thiol-thione | Add full Sitzmann/SIE rule table |
| **Canonical SMILES** | Internally consistent but not bit-for-bit identical to RDKit's InChI-derived canonical form | Add RDKit-compatible mode via MolBlock round-trip |
| **Stereo chemistry** | Parsed but not stored or propagated to 3D | Add CIP ranking and stereo-aware embedding |
| **Implicit H in AMW** | Only explicit (bracket) H count used; organic-subset atom H not added | Add valence-based implicit H calculation |
| **Rotatable bonds** | Bridge-detection BFS is O(bonds^2); slow on molecules >50 heavy atoms | Precompute adjacency list |
| **3D embedding** | Heavy-atom-only distance geometry; no torsional statistics (vs ETKDG) | Add torsion angle statistics |
| **Macromolecules** | No support for BIOVIA extended SMILES or HELM | Out of scope for v0.1 |
| **Reaction SMILES** | Not supported | Out of scope |
| **SMARTS queries** | `has_substruct_match` uses SMILES queries, not SMARTS patterns | Add SMARTS parser |

