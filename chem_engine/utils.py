"""
Serialization helpers for RDKit <-> RustMolecule conversions

Note on PyO3 property naming:
  #[getter] get_X exposes property as `x` (lowercase, without 'get_' prefix).
  e.g. get_num_atoms → .num_atoms
       get_coords_2d → .coords_2d
"""
from ._rust import RustMolecule, Atom, Bond, BondType


def to_rdkit(rust_mol: RustMolecule):
    """
    Converts a RustMolecule to a native RDKit Chem.Mol object.
    Preserves all atoms, bonds, and 2D/3D coordinates.

    Uses V2000 MolBlock as the interchange format
    Returns None if RDKit is not installed.
    """
    from rdkit import Chem

    num_atoms = rust_mol.num_atoms
    num_bonds = rust_mol.num_bonds

    # Build a standard V2000 MolBlock
    lines = ["", "", ""]  # 3-line header

    # Counts line: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    lines.append(f"{num_atoms:3d}{num_bonds:3d}  0  0  0  0  0  0  0  0999 V2000")

    # Get coordinate arrays (properties, not method calls)
    coords_3d = rust_mol.coords_3d
    coords_2d = rust_mol.coords_2d

    # Atom block
    for i in range(num_atoms):
        atom = rust_mol.get_atom(i)

        if coords_3d is not None and i < len(coords_3d):
            x, y, z = coords_3d[i]
        elif coords_2d is not None and i < len(coords_2d):
            x, y = coords_2d[i]
            z = 0.0
        else:
            x, y, z = 0.0, 0.0, 0.0

        symbol = atom.symbol
        lines.append(
            f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )

    # Bond block
    for j in range(num_bonds):
        bond = rust_mol.get_bond(j)
        u = bond.source_idx + 1   # MolBlock uses 1-based indexing
        v = bond.target_idx + 1

        b_type = 1  # default: single
        if bond.bond_type == BondType.Double:
            b_type = 2
        elif bond.bond_type == BondType.Triple:
            b_type = 3
        elif bond.bond_type == BondType.Aromatic:
            b_type = 4

        lines.append(f"{u:3d}{v:3d}{b_type:3d}  0  0  0  0")

    lines.append("M  END")
    mol_block = "\n".join(lines)

    rd_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    return rd_mol


def from_rdkit(rdkit_mol) -> RustMolecule:
    """
    Converts a native RDKit Chem.Mol object to a RustMolecule.
    Preserves all atoms, bonds, and 2D/3D coordinates.
    """
    from rdkit import Chem

    rust_mol = RustMolecule()
    num_atoms = rdkit_mol.GetNumAtoms()

    # Detect conformer coordinates
    coords = None
    is_3d = False
    if rdkit_mol.GetNumConformers() > 0:
        conf = rdkit_mol.GetConformer()
        is_3d = conf.Is3D()
        coords = [list(conf.GetAtomPosition(i)) for i in range(num_atoms)]

    # Copy atoms
    for i in range(num_atoms):
        rd_atom = rdkit_mol.GetAtomWithIdx(i)
        atomic_num = rd_atom.GetAtomicNum()
        charge = rd_atom.GetFormalCharge()
        num_hs = rd_atom.GetTotalNumHs()
        is_aromatic = rd_atom.GetIsAromatic()

        atom = Atom(atomic_num, charge, num_hs, is_aromatic)
        rust_mol.add_atom(atom)

    # Copy bonds
    for rd_bond in rdkit_mol.GetBonds():
        u = rd_bond.GetBeginAtomIdx()
        v = rd_bond.GetEndAtomIdx()
        rd_type = rd_bond.GetBondType()

        b_type = BondType.Single
        if rd_type == Chem.BondType.DOUBLE:
            b_type = BondType.Double
        elif rd_type == Chem.BondType.TRIPLE:
            b_type = BondType.Triple
        elif rd_type == Chem.BondType.AROMATIC:
            b_type = BondType.Aromatic

        rust_mol.add_bond(u, v, b_type)

    # Apply coordinates
    if coords is not None:
        if is_3d:
            rust_mol.coords_3d = coords
        else:
            rust_mol.coords_2d = [c[:2] for c in coords]

    return rust_mol
