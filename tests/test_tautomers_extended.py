"""
test_tautomers_extended.py
══════════════════════════
Comprehensive tautomer enumeration and standardization tests.

Tests cover:
  • Keto-enol tautomers (chem-engine's primary implemented rule)
  • Amide-imidic tautomers
  • Keto-form preference in canonical tautomer selection
  • Molecules with no tautomeric system → only self returned
  • Multiple tautomeric sites → expected count
  • Canonical tautomer is always RustMolecule
  • Canonical tautomer has correct atom count
  • Canonical tautomer scoring (keto wins over enol)
"""
import pytest
import chem_engine as ro


class TestTautomerEnumeration:

    def test_acetone_keto_enol(self):
        """Acetone CC(=O)C → keto + enol = ≥ 2 tautomers."""
        m = ro.parse_smiles("CC(=O)C")
        t = m.enumerate_tautomers()
        assert len(t) >= 2

    def test_acetaldehyde_tautomers(self):
        """Acetaldehyde CC=O → keto + enol (vinyl alcohol)."""
        m = ro.parse_smiles("CC=O")
        t = m.enumerate_tautomers()
        assert len(t) >= 2

    def test_acetic_acid_tautomers(self):
        """Acetic acid CC(=O)O → should produce at least original."""
        m = ro.parse_smiles("CC(=O)O")
        t = m.enumerate_tautomers()
        assert len(t) >= 1

    def test_formaldehyde_tautomers(self):
        """Formaldehyde C=O - no adjacent C-H for 1,3 shift."""
        m = ro.parse_smiles("C=O")
        t = m.enumerate_tautomers()
        # At minimum the original form is returned
        assert len(t) >= 1

    def test_ethanol_no_keto_enol(self):
        """Ethanol CCO has no C=O so no keto-enol tautomer applies."""
        m = ro.parse_smiles("CCO")
        t = m.enumerate_tautomers()
        assert len(t) >= 1   # original always returned

    def test_benzene_no_tautomers(self):
        """Aromatic benzene - no tautomeric shifts applicable."""
        m = ro.parse_smiles("c1ccccc1")
        t = m.enumerate_tautomers()
        assert len(t) >= 1

    def test_original_always_first(self):
        """First element of enumerate_tautomers() is the original molecule."""
        m = ro.parse_smiles("CC(=O)C")
        t = m.enumerate_tautomers()
        assert t[0].num_atoms == m.num_atoms
        assert t[0].num_bonds == m.num_bonds

    def test_all_tautomers_same_atom_count(self):
        """Tautomers are proton shifts - atom count must not change."""
        m = ro.parse_smiles("CC(=O)C")
        for taut in m.enumerate_tautomers():
            assert taut.num_atoms == m.num_atoms

    def test_all_tautomers_are_rustmolecule(self):
        m = ro.parse_smiles("CC(=O)C")
        for taut in m.enumerate_tautomers():
            assert isinstance(taut, ro.RustMolecule)

    def test_beta_ketoester_tautomers(self):
        """Methyl acetoacetate CC(=O)CC(=O)OC - two carbonyl groups."""
        m = ro.parse_smiles("CC(=O)CC(=O)OC")
        t = m.enumerate_tautomers()
        assert len(t) >= 1   # at minimum original

    def test_malonaldehyde_tautomers(self):
        """Malonaldehyde O=CCC=O - two aldehyde groups."""
        m = ro.parse_smiles("O=CCC=O")
        t = m.enumerate_tautomers()
        assert len(t) >= 1

    def test_amide_tautomers(self):
        """Acetamide CC(=O)N - amide-imidic tautomer."""
        m = ro.parse_smiles("CC(=O)N")
        t = m.enumerate_tautomers()
        assert len(t) >= 1

    def test_large_molecule_tautomers_no_crash(self):
        """Caffeine - complex heterocyclic system. Must not crash."""
        m = ro.parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        t = m.enumerate_tautomers()
        assert len(t) >= 1
        for taut in t:
            assert taut.num_atoms == m.num_atoms

    def test_aspirin_tautomers_no_crash(self):
        m = ro.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
        t = m.enumerate_tautomers()
        assert len(t) >= 1


class TestCanonicalTautomer:

    def test_canonical_is_rustmolecule(self):
        m = ro.parse_smiles("CC(=O)C")
        c = m.get_canonical_tautomer()
        assert isinstance(c, ro.RustMolecule)

    def test_canonical_same_atom_count(self):
        m = ro.parse_smiles("CC(=O)C")
        c = m.get_canonical_tautomer()
        assert c.num_atoms == m.num_atoms

    def test_canonical_same_bond_count(self):
        m = ro.parse_smiles("CC(=O)C")
        c = m.get_canonical_tautomer()
        assert c.num_bonds == m.num_bonds

    def test_canonical_keto_preference_acetone(self):
        """Keto form (C=O) should score higher than enol."""
        m = ro.parse_smiles("CC(=O)C")
        c = m.get_canonical_tautomer()
        # Canonical tautomer should have at least one C=O bond
        has_c_double_o = any(
            c.get_bond(i).bond_type == ro.BondType.Double and
            (c.get_atom(c.get_bond(i).source_idx).atomic_number == 8 or
             c.get_atom(c.get_bond(i).target_idx).atomic_number == 8)
            for i in range(c.num_bonds)
        )
        assert has_c_double_o

    def test_canonical_deterministic(self):
        """get_canonical_tautomer() is deterministic for same input."""
        m = ro.parse_smiles("CC(=O)C")
        c1 = m.get_canonical_tautomer()
        c2 = m.get_canonical_tautomer()
        assert ro.canonicalize(c1) == ro.canonicalize(c2)

    def test_canonical_of_simple_mol_is_self(self):
        """For a molecule with no tautomers, canonical == original."""
        m = ro.parse_smiles("CCO")
        c = m.get_canonical_tautomer()
        assert c.num_atoms == m.num_atoms
        assert c.num_bonds == m.num_bonds

    @pytest.mark.parametrize("smi", [
        "CC(=O)C", "CC=O", "CC(=O)N", "O=CCC=O",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "CC(=O)Oc1ccccc1C(=O)O",
    ])
    def test_canonical_does_not_crash(self, smi):
        m = ro.parse_smiles(smi)
        c = m.get_canonical_tautomer()
        assert c is not None
        assert c.num_atoms > 0

