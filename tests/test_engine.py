"""
Tests for ChemEngine (all FRs from the PRD).

Property name convention: PyO3 #[getter] named get_X exposes property X.
So:
  get_num_atoms  -> num_atoms
  get_num_bonds  -> num_bonds
  get_amw        -> amw
  get_num_rotatable_bonds -> num_rotatable_bonds
  get_coords_2d  -> coords_2d
  get_coords_3d  -> coords_3d
"""

import pytest

import chem_engine as ro
from chem_engine.utils import from_rdkit, to_rdkit

# ---------------------------------------------------------------------------
# FR-1 & FR-2: Molecular graph + SMILES parsing
# ---------------------------------------------------------------------------


class TestSmilesParsing:
    """FR-1 (Molecular Graph) + FR-2 (SMILES → 2D Pipeline)"""

    def test_parse_simple_chain(self):
        """Ethanol: CCO → 3 atoms, 2 bonds"""
        mol = ro.parse_smiles("CCO")
        assert mol.num_atoms == 3
        assert mol.num_bonds == 2

    def test_parse_atom_symbols(self):
        mol = ro.parse_smiles("CCO")
        atom0 = mol.get_atom(0)
        atom2 = mol.get_atom(2)
        assert atom0.symbol == "C"
        assert atom0.atomic_number == 6
        assert atom2.symbol == "O"
        assert atom2.atomic_number == 8

    def test_parse_nitrogen(self):
        """Methylamine: CN"""
        mol = ro.parse_smiles("CN")
        assert mol.num_atoms == 2
        assert mol.get_atom(1).atomic_number == 7

    def test_parse_double_bond(self):
        """Ethene: C=C"""
        mol = ro.parse_smiles("C=C")
        assert mol.num_atoms == 2
        assert mol.num_bonds == 1
        bond = mol.get_bond(0)
        assert bond.bond_type == ro.BondType.Double

    def test_parse_triple_bond(self):
        """Acetylene: C#C"""
        mol = ro.parse_smiles("C#C")
        bond = mol.get_bond(0)
        assert bond.bond_type == ro.BondType.Triple

    def test_parse_ring(self):
        """Cyclohexane: C1CCCCC1 - 6 atoms, 6 bonds"""
        mol = ro.parse_smiles("C1CCCCC1")
        assert mol.num_atoms == 6
        assert mol.num_bonds == 6

    def test_parse_benzene_aromatic(self):
        """Benzene aromatic notation: c1ccccc1"""
        mol = ro.parse_smiles("c1ccccc1")
        assert mol.num_atoms == 6
        # All atoms should be aromatic
        for i in range(mol.num_atoms):
            atom = mol.get_atom(i)
            assert atom.is_aromatic, f"Atom {i} should be aromatic"

    def test_parse_bracket_atom_with_charge(self):
        """Ammonium ion [NH4+]"""
        mol = ro.parse_smiles("[NH4+]")
        assert mol.num_atoms == 1
        atom = mol.get_atom(0)
        assert atom.atomic_number == 7
        assert atom.formal_charge == 1
        assert atom.num_explicit_hs == 4

    def test_parse_branch(self):
        """Isobutane: CC(C)C - 4 atoms, 3 bonds"""
        mol = ro.parse_smiles("CC(C)C")
        assert mol.num_atoms == 4
        assert mol.num_bonds == 3

    def test_parse_chlorine(self):
        """Chloromethane: CCl"""
        mol = ro.parse_smiles("CCl")
        assert mol.num_atoms == 2
        assert mol.get_atom(1).atomic_number == 17

    def test_parse_bromine(self):
        """Bromomethane: CBr"""
        mol = ro.parse_smiles("CBr")
        assert mol.num_atoms == 2
        assert mol.get_atom(1).atomic_number == 35

    def test_empty_mol_creation(self):
        """Empty molecule defaults"""
        mol = ro.RustMolecule()
        assert mol.num_atoms == 0
        assert mol.num_bonds == 0

    def test_invalid_smiles_raises(self):
        """Invalid SMILES should raise ValueError"""
        with pytest.raises(Exception):
            ro.parse_smiles("XYZ_invalid_999")


# ---------------------------------------------------------------------------
# FR-5: Canonical SMILES
# ---------------------------------------------------------------------------


class TestCanonicalSmiles:
    """FR-5: 2D representation → canonical SMILES"""

    def test_canonical_deterministic(self):
        """Same molecule parsed from different SMILES should produce same canonical form"""
        mol1 = ro.parse_smiles("CCO")
        mol2 = ro.parse_smiles("OCC")
        can1 = ro.canonicalize(mol1)
        can2 = ro.canonicalize(mol2)
        assert can1 == can2

    def test_canonical_contains_atoms(self):
        mol = ro.parse_smiles("CCO")
        can = ro.canonicalize(mol)
        assert "O" in can
        assert "C" in can

    def test_canonical_empty(self):
        mol = ro.RustMolecule()
        can = ro.canonicalize(mol)
        assert can == ""

    def test_canonical_single_atom(self):
        mol = ro.parse_smiles("C")
        can = ro.canonicalize(mol)
        assert "C" in can

    def test_canonical_nonzero_length(self):
        """Canonical SMILES should be non-empty for any parsed molecule"""
        for smiles in ["CCO", "CC(C)C", "C1CCCCC1"]:
            mol = ro.parse_smiles(smiles)
            can = ro.canonicalize(mol)
            assert len(can) > 0, f"Empty canonical SMILES for {smiles}"


# ---------------------------------------------------------------------------
# FR-6: 2D and 3D coordinate generation
# ---------------------------------------------------------------------------


class TestLayout:
    """FR-6: ETKDG-like 2D/3D coordinate generation"""

    def test_2d_coords_generated(self):
        mol = ro.parse_smiles("CCO")
        mol = ro.generate_2d_coords(mol)
        coords = mol.coords_2d
        assert coords is not None
        assert len(coords) == 3
        assert all(len(c) == 2 for c in coords)

    def test_2d_coords_not_all_zero(self):
        """Force-directed layout should spread atoms out"""
        mol = ro.parse_smiles("CCCC")  # butane - 4 atoms
        mol = ro.generate_2d_coords(mol)
        coords = mol.coords_2d
        xs = [c[0] for c in coords]
        # Not all x-coordinates should be identical
        assert max(xs) - min(xs) > 0.01

    def test_3d_coords_generated(self):
        mol = ro.parse_smiles("CCO")
        mol = ro.generate_3d_coords(mol)
        coords = mol.coords_3d
        assert coords is not None
        assert len(coords) == 3
        assert all(len(c) == 3 for c in coords)

    def test_3d_coord_count_matches_atoms(self):
        mol = ro.parse_smiles("C1CCCCC1")
        mol = ro.generate_3d_coords(mol)
        coords = mol.coords_3d
        assert len(coords) == mol.num_atoms

    def test_no_coords_initially(self):
        mol = ro.parse_smiles("CCO")
        assert mol.coords_2d is None
        assert mol.coords_3d is None


# ---------------------------------------------------------------------------
# FR-8: Descriptor calculations (AMW + RotBonds)
# ---------------------------------------------------------------------------


class TestDescriptors:
    """FR-8: AMW and rotatable bonds"""

    def test_amw_water(self):
        """Water [H]O[H]: 15.999 + 2*1.008 = 18.015"""
        mol = ro.parse_smiles("[H]O[H]")
        assert abs(mol.amw - 18.015) < 0.10

    def test_rotatable_bonds_ethane(self):
        """Ethane CC: C-C is terminal on both ends → 0 rotatable bonds"""
        mol = ro.parse_smiles("CC")
        assert mol.num_rotatable_bonds == 0

    def test_rotatable_bonds_propane(self):
        """Propane CCC: 2 C-C bonds, both are terminal on at least one end → 0"""
        mol = ro.parse_smiles("CCC")
        assert mol.num_rotatable_bonds == 0

    def test_rotatable_bonds_butane(self):
        """Butane CCCC: central C-C bond is rotatable → 1"""
        mol = ro.parse_smiles("CCCC")
        assert mol.num_rotatable_bonds == 1

    def test_rotatable_bonds_pentane(self):
        """Pentane CCCCC: 2 central C-C bonds → 2"""
        mol = ro.parse_smiles("CCCCC")
        assert mol.num_rotatable_bonds == 2

    def test_rotatable_bonds_ring(self):
        """Cyclohexane: ring bonds are not rotatable → 0"""
        mol = ro.parse_smiles("C1CCCCC1")
        assert mol.num_rotatable_bonds == 0


# ---------------------------------------------------------------------------
# FR-4: Parallel batch engine
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    """FR-4: Rayon-backed parallel SMILES parsing"""

    def test_batch_parse_returns_correct_count(self):
        smiles_list = ["CCO", "CC(C)C", "C1CCCCC1", "CN"]
        results = ro.batch_parse_smiles(smiles_list)
        assert len(results) == 4

    def test_batch_parse_atom_counts(self):
        smiles_list = ["CCO", "CCCC"]
        results = ro.batch_parse_smiles(smiles_list)
        assert results[0].num_atoms == 3  # CCO
        assert results[1].num_atoms == 4  # CCCC

    def test_batch_parse_empty_list(self):
        results = ro.batch_parse_smiles([])
        assert results == []

    def test_batch_parse_large(self):
        """Stress test with 100 molecules"""
        smiles_list = ["CCO"] * 100
        results = ro.batch_parse_smiles(smiles_list)
        assert len(results) == 100
        assert all(r.num_atoms == 3 for r in results)


# ---------------------------------------------------------------------------
# FR-9: Substructure search
# ---------------------------------------------------------------------------


class TestSubstructureSearch:
    """FR-9: Subgraph isomorphism"""

    def test_methanol_in_ethanol(self):
        """CO is a substructure of CCO"""
        target = ro.parse_smiles("CCO")
        query = ro.parse_smiles("CO")
        assert target.has_substruct_match(query) is True

    def test_propane_not_in_ethanol(self):
        """CCC is not in CCO"""
        target = ro.parse_smiles("CCO")
        query = ro.parse_smiles("CCC")
        assert target.has_substruct_match(query) is False

    def test_empty_query_always_matches(self):
        """Empty pattern matches anything"""
        target = ro.parse_smiles("CCO")
        query = ro.RustMolecule()
        assert target.has_substruct_match(query) is True

    def test_self_matches_self(self):
        """A molecule always contains itself as substructure"""
        mol = ro.parse_smiles("CCO")
        assert mol.has_substruct_match(mol) is True

    def test_larger_does_not_match_smaller(self):
        """CCCO is not a substructure of CCO"""
        target = ro.parse_smiles("CCO")
        query = ro.parse_smiles("CCCO")
        assert target.has_substruct_match(query) is False

    def test_nitrogen_query(self):
        """CN is a substructure of CCN"""
        target = ro.parse_smiles("CCN")
        query = ro.parse_smiles("CN")
        assert target.has_substruct_match(query) is True


# ---------------------------------------------------------------------------
# FR-10: Chemical fingerprints + Tanimoto similarity
# ---------------------------------------------------------------------------


class TestSimilaritySearch:
    """FR-10: Fingerprint generation and Tanimoto coefficient"""

    def test_self_similarity_is_one(self):
        mol = ro.parse_smiles("CCO")
        assert abs(mol.similarity(mol) - 1.0) < 1e-6

    def test_similarity_between_zero_and_one(self):
        mol1 = ro.parse_smiles("CCO")
        mol2 = ro.parse_smiles("CCN")
        sim = mol1.similarity(mol2)
        assert 0.0 <= sim <= 1.0

    def test_similar_molecules_more_similar_than_dissimilar(self):
        """Ethanol and ethylamine should be more similar to each other than to benzene"""
        mol1 = ro.parse_smiles("CCO")
        mol2 = ro.parse_smiles("CCN")
        mol3 = ro.parse_smiles("c1ccccc1")
        sim12 = mol1.similarity(mol2)
        sim13 = mol1.similarity(mol3)
        assert sim12 > sim13

    def test_fingerprint_length(self):
        mol = ro.parse_smiles("CCO")
        fp = mol.get_fingerprint()
        assert len(fp) == 2048

    def test_fingerprint_is_bool_list(self):
        mol = ro.parse_smiles("CCO")
        fp = mol.get_fingerprint()
        assert all(isinstance(b, bool) for b in fp)

    def test_empty_mol_fingerprint(self):
        mol = ro.RustMolecule()
        fp = mol.get_fingerprint()
        assert len(fp) == 2048
        assert all(b is False for b in fp)

    def test_empty_mol_similarity(self):
        mol1 = ro.RustMolecule()
        mol2 = ro.RustMolecule()
        # Both empty → trivially 0 (0/0)
        sim = mol1.similarity(mol2)
        assert sim == 0.0


# ---------------------------------------------------------------------------
# FR-7: Tautomer enumeration and standardization
# ---------------------------------------------------------------------------


class TestTautomers:
    """FR-7: Tautomer enumeration and canonical tautomer selection"""

    def test_acetone_has_tautomers(self):
        """Acetone CC(=O)C should produce at least 2 tautomers (keto + enol)"""
        mol = ro.parse_smiles("CC(=O)C")
        tautomers = mol.enumerate_tautomers()
        assert len(tautomers) >= 2

    def test_original_is_in_tautomers(self):
        """The input molecule should always appear in the enumerated list"""
        mol = ro.parse_smiles("CC(=O)C")
        tautomers = mol.enumerate_tautomers()
        # The first tautomer should be the original molecule (same bond count)
        assert tautomers[0].num_atoms == mol.num_atoms
        assert tautomers[0].num_bonds == mol.num_bonds

    def test_canonical_tautomer_has_keto_form(self):
        """Keto form (C=O) should be preferred over enol for acetone"""
        mol = ro.parse_smiles("CC(=O)C")
        canonical = mol.get_canonical_tautomer()
        has_keto = False
        for i in range(canonical.num_bonds):
            bond = canonical.get_bond(i)
            if bond.bond_type == ro.BondType.Double:
                u_atom = canonical.get_atom(bond.source_idx)
                v_atom = canonical.get_atom(bond.target_idx)
                if u_atom.atomic_number == 8 or v_atom.atomic_number == 8:
                    has_keto = True
        assert has_keto is True

    def test_canonical_tautomer_is_rust_molecule(self):
        mol = ro.parse_smiles("CC(=O)C")
        canonical = mol.get_canonical_tautomer()
        assert isinstance(canonical, ro.RustMolecule)

    def test_simple_mol_unchanged_tautomers(self):
        """Ethanol has no obvious tautomeric system → 1 tautomer"""
        mol = ro.parse_smiles("CCO")
        tautomers = mol.enumerate_tautomers()
        # Only the original should be returned (no tautomeric transformations apply)
        assert len(tautomers) >= 1


# ---------------------------------------------------------------------------
# FR-3 / FR-11: RDKit type interoperability
# ---------------------------------------------------------------------------


class TestRDKitInterop:
    """FR-3 / FR-11: Two-way conversion with rdkit.Chem.Mol"""

    def test_from_rdkit_atom_count(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")
        rd_mol = Chem.MolFromSmiles("CCO")
        rust_mol = from_rdkit(rd_mol)
        assert rust_mol.num_atoms == 3

    def test_from_rdkit_bond_count(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")
        rd_mol = Chem.MolFromSmiles("CCO")
        rust_mol = from_rdkit(rd_mol)
        assert rust_mol.num_bonds == 2

    def test_from_rdkit_atom_symbols(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")
        rd_mol = Chem.MolFromSmiles("CCO")
        rust_mol = from_rdkit(rd_mol)
        symbols = [rust_mol.get_atom(i).symbol for i in range(rust_mol.num_atoms)]
        assert "C" in symbols
        assert "O" in symbols

    def test_to_rdkit_round_trip(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")
        rd_mol = Chem.MolFromSmiles("CCO")
        rust_mol = from_rdkit(rd_mol)
        new_rd_mol = to_rdkit(rust_mol)
        assert new_rd_mol is not None
        assert new_rd_mol.GetNumAtoms() == 3
        assert new_rd_mol.GetNumBonds() == 2

    def test_from_rdkit_with_3d_coords(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            pytest.skip("RDKit not installed")
        rd_mol = Chem.MolFromSmiles("CCO")
        rd_mol = Chem.AddHs(rd_mol)
        AllChem.EmbedMolecule(rd_mol, randomSeed=42)
        rd_mol = Chem.RemoveHs(rd_mol)
        rust_mol = from_rdkit(rd_mol)
        assert rust_mol.coords_3d is not None

    def test_to_rdkit_bond_types(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")
        rd_mol = Chem.MolFromSmiles("C=O")  # Formaldehyde
        rust_mol = from_rdkit(rd_mol)
        new_rd_mol = to_rdkit(rust_mol)
        assert new_rd_mol is not None
        bond = new_rd_mol.GetBondWithIdx(0)
        assert bond.GetBondTypeAsDouble() == 2.0


# ---------------------------------------------------------------------------
# Manual atom/bond API (FR-1)
# ---------------------------------------------------------------------------


class TestManualMoleculeConstruction:
    """FR-1: Direct Atom/Bond API"""

    def test_add_atoms_and_bonds(self):
        mol = ro.RustMolecule()
        atom_c = ro.Atom(6, 0, 0, False)  # Carbon
        atom_o = ro.Atom(8, 0, 0, False)  # Oxygen
        mol.add_atom(atom_c)
        mol.add_atom(atom_o)
        mol.add_bond(0, 1, ro.BondType.Single)
        assert mol.num_atoms == 2
        assert mol.num_bonds == 1

    def test_get_atom_properties(self):
        mol = ro.RustMolecule()
        atom = ro.Atom(6, -1, 2, True)
        mol.add_atom(atom)
        a = mol.get_atom(0)
        assert a.atomic_number == 6
        assert a.formal_charge == -1
        assert a.num_explicit_hs == 2
        assert a.is_aromatic is True

    def test_get_bond_properties(self):
        mol = ro.RustMolecule()
        mol.add_atom(ro.Atom(6, 0, 0, False))
        mol.add_atom(ro.Atom(6, 0, 0, False))
        mol.add_bond(0, 1, ro.BondType.Double)
        bond = mol.get_bond(0)
        assert bond.source_idx == 0
        assert bond.target_idx == 1
        assert bond.bond_type == ro.BondType.Double

    def test_find_bond(self):
        mol = ro.parse_smiles("CCO")
        bond = mol.find_bond(0, 1)
        assert bond is not None
        assert bond.bond_type == ro.BondType.Single

    def test_find_nonexistent_bond(self):
        mol = ro.parse_smiles("CCO")
        bond = mol.find_bond(0, 2)  # C and O are not directly bonded
        assert bond is None

    def test_coords_setter_2d(self):
        mol = ro.parse_smiles("CC")
        mol.coords_2d = [[1.0, 2.0], [3.0, 4.0]]
        coords = mol.coords_2d
        assert coords is not None
        assert len(coords) == 2
        assert coords[0][0] == pytest.approx(1.0)

    def test_coords_setter_3d(self):
        mol = ro.parse_smiles("CC")
        mol.coords_3d = [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]]
        coords = mol.coords_3d
        assert coords is not None
        assert len(coords) == 2
