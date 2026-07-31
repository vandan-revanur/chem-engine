"""
test_invariants.py
══════════════════
Mathematical and chemical invariants that MUST hold for any correct
cheminformatics implementation.

Every test here verifies a property that is true by definition,
independent of implementation details:
  • Tanimoto coefficient range and symmetry
  • Fingerprint properties (length, idempotency)
  • Canonical SMILES idempotency
  • Substructure transitivity and self-containment
  • Coordinate count equals atom count
  • AMW ordering invariants
"""
import math
import pytest
import chem_engine as ro

REFERENCE_SMILES = [
    "CCO", "CC(C)C", "C1CCCCC1", "c1ccccc1",
    "CC(=O)O", "CN", "CCN", "CC(=O)N",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",   # caffeine
]


# ─── Tanimoto / Similarity invariants ────────────────────────────────────────

class TestSimilarityInvariants:

    def test_self_similarity_always_one(self):
        """sim(A, A) == 1.0 for any non-empty molecule."""
        for smi in REFERENCE_SMILES:
            m = ro.parse_smiles(smi)
            assert abs(m.similarity(m) - 1.0) < 1e-9, \
                f"Self-similarity != 1 for {smi}"

    def test_similarity_range_0_to_1(self):
        """0 ≤ sim(A, B) ≤ 1 for all molecule pairs."""
        mols = [ro.parse_smiles(s) for s in REFERENCE_SMILES]
        for i, a in enumerate(mols):
            for j, b in enumerate(mols):
                s = a.similarity(b)
                assert 0.0 <= s <= 1.0, \
                    f"Similarity out of range [{i},{j}]: {s}"

    def test_similarity_symmetric(self):
        """sim(A, B) == sim(B, A)."""
        mols = [ro.parse_smiles(s) for s in REFERENCE_SMILES[:6]]
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                s_ij = mols[i].similarity(mols[j])
                s_ji = mols[j].similarity(mols[i])
                assert abs(s_ij - s_ji) < 1e-9, \
                    f"Similarity not symmetric at [{i},{j}]: {s_ij} vs {s_ji}"

    def test_similar_more_than_dissimilar(self):
        """Ethanol/ethylamine should be more similar to each other than to naphthalene."""
        ethanol    = ro.parse_smiles("CCO")
        ethylamine = ro.parse_smiles("CCN")
        naphthalene = ro.parse_smiles("c1ccc2ccccc2c1")
        assert ethanol.similarity(ethylamine) > ethanol.similarity(naphthalene)

    def test_empty_mol_similarity_zero(self):
        empty = ro.RustMolecule()
        real  = ro.parse_smiles("CCO")
        assert empty.similarity(real) == 0.0
        assert real.similarity(empty) == 0.0

    def test_structurally_different_low_similarity(self):
        """Cyclohexane and naphthalene should have different fingerprints."""
        cyclohexane = ro.parse_smiles("C1CCCCC1")
        naphthalene = ro.parse_smiles("c1ccc2ccccc2c1")
        assert cyclohexane.similarity(naphthalene) < 1.0


class TestFingerprintInvariants:

    def test_fingerprint_length_always_2048(self):
        for smi in REFERENCE_SMILES + [""]:
            try:
                m = ro.parse_smiles(smi) if smi else ro.RustMolecule()
            except Exception:
                m = ro.RustMolecule()
            fp = m.get_fingerprint()
            assert len(fp) == 2048, f"FP length != 2048 for {smi!r}"

    def test_fingerprint_idempotent(self):
        """get_fingerprint() called twice returns the same bits."""
        for smi in REFERENCE_SMILES[:5]:
            m = ro.parse_smiles(smi)
            fp1 = m.get_fingerprint()
            fp2 = m.get_fingerprint()
            assert fp1 == fp2, f"Fingerprint not idempotent for {smi}"

    def test_fingerprint_not_all_false_for_real_molecule(self):
        for smi in REFERENCE_SMILES:
            m = ro.parse_smiles(smi)
            fp = m.get_fingerprint()
            assert any(fp), f"Fingerprint all-false for {smi}"

    def test_fingerprint_bool_type(self):
        m = ro.parse_smiles("CCO")
        fp = m.get_fingerprint()
        assert all(isinstance(b, bool) for b in fp)

    def test_different_molecules_different_fingerprints(self):
        """Benzene and cyclohexane have structurally different fingerprints."""
        benz = ro.parse_smiles("c1ccccc1")
        chex = ro.parse_smiles("C1CCCCC1")
        assert benz.get_fingerprint() != chex.get_fingerprint()

    def test_same_molecule_same_fingerprint(self):
        """Parsing the same SMILES twice → identical fingerprints."""
        m1 = ro.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
        m2 = ro.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert m1.get_fingerprint() == m2.get_fingerprint()


# ─── Canonical SMILES invariants ─────────────────────────────────────────────

class TestCanonicalSmilesInvariants:

    def test_idempotent_on_all_reference_mols(self):
        for smi in REFERENCE_SMILES:
            m1 = ro.parse_smiles(smi)
            c1 = ro.canonicalize(m1)
            m2 = ro.parse_smiles(c1)
            c2 = ro.canonicalize(m2)
            assert c1 == c2, f"Not idempotent for {smi}: {c1!r} vs {c2!r}"

    def test_canonical_nonempty_for_nonempty_mol(self):
        for smi in REFERENCE_SMILES:
            m = ro.parse_smiles(smi)
            assert len(ro.canonicalize(m)) > 0

    def test_canonical_empty_for_empty_mol(self):
        m = ro.RustMolecule()
        assert ro.canonicalize(m) == ""

    def test_canonical_same_for_same_structure(self):
        """CCO and OCC should canonicalize to the same string."""
        c1 = ro.canonicalize(ro.parse_smiles("CCO"))
        c2 = ro.canonicalize(ro.parse_smiles("OCC"))
        assert c1 == c2

    def test_canonical_different_for_different_structure(self):
        """Ethanol and methanol are not the same."""
        c1 = ro.canonicalize(ro.parse_smiles("CCO"))
        c2 = ro.canonicalize(ro.parse_smiles("CO"))
        assert c1 != c2


# ─── Substructure invariants ─────────────────────────────────────────────────

class TestSubstructureInvariants:

    def test_every_molecule_contains_itself(self):
        for smi in REFERENCE_SMILES:
            m = ro.parse_smiles(smi)
            assert m.has_substruct_match(m), \
                f"Molecule does not contain itself: {smi}"

    def test_every_molecule_contains_empty_query(self):
        query = ro.RustMolecule()
        for smi in REFERENCE_SMILES:
            m = ro.parse_smiles(smi)
            assert m.has_substruct_match(query), \
                f"Empty query not matched by {smi}"

    def test_empty_contains_empty(self):
        assert ro.RustMolecule().has_substruct_match(ro.RustMolecule()) is True

    def test_empty_does_not_contain_real_molecule(self):
        empty = ro.RustMolecule()
        real  = ro.parse_smiles("CCO")
        assert empty.has_substruct_match(real) is False

    def test_substructure_transitivity(self):
        """If A ⊆ B and B ⊆ C, then A ⊆ C."""
        a = ro.parse_smiles("CO")       # methanol
        b = ro.parse_smiles("CCO")      # ethanol
        c = ro.parse_smiles("CCCO")     # propanol
        assert b.has_substruct_match(a)
        assert c.has_substruct_match(b)
        assert c.has_substruct_match(a)

    def test_asymmetry_non_substructure(self):
        """If A ⊄ B it does not follow that B ⊄ A (both can be true)."""
        big = ro.parse_smiles("c1ccccc1")   # benzene
        small = ro.parse_smiles("CC")       # ethane
        # ethane is NOT a substructure of benzene (no saturated C-C single in benzene)
        assert not big.has_substruct_match(small)


# ─── Coordinate invariants ────────────────────────────────────────────────────

class TestCoordinateInvariants:

    def test_2d_coord_count_equals_atom_count(self):
        for smi in REFERENCE_SMILES:
            m = ro.generate_2d_coords(ro.parse_smiles(smi))
            assert len(m.coords_2d) == m.num_atoms, \
                f"2D coord count != atom count for {smi}"

    def test_3d_coord_count_equals_atom_count(self):
        for smi in REFERENCE_SMILES:
            m = ro.generate_3d_coords(ro.parse_smiles(smi))
            assert len(m.coords_3d) == m.num_atoms, \
                f"3D coord count != atom count for {smi}"

    def test_2d_coord_dimension_is_2(self):
        m = ro.generate_2d_coords(ro.parse_smiles("CCO"))
        for c in m.coords_2d:
            assert len(c) == 2

    def test_3d_coord_dimension_is_3(self):
        m = ro.generate_3d_coords(ro.parse_smiles("CCO"))
        for c in m.coords_3d:
            assert len(c) == 3

    def test_2d_layout_spreads_atoms(self):
        """Force-directed layout should not collapse all atoms to one point."""
        m = ro.generate_2d_coords(ro.parse_smiles("CCCCCC"))  # hexane
        xs = [c[0] for c in m.coords_2d]
        ys = [c[1] for c in m.coords_2d]
        assert max(xs) - min(xs) > 0.01 or max(ys) - min(ys) > 0.01

    def test_3d_layout_spreads_atoms(self):
        m = ro.generate_3d_coords(ro.parse_smiles("CCCCCC"))
        coords = m.coords_3d
        # At least one dimension should vary
        for dim in range(3):
            vals = [c[dim] for c in coords]
            if max(vals) - min(vals) > 0.01:
                return  # found spread in at least one dimension
        pytest.fail("3D layout did not spread atoms in any dimension")

    def test_3d_bonded_atom_distance_reasonable(self):
        """Adjacent bonded atoms should be 0.5-4.0 Å apart (rough check)."""
        m = ro.generate_3d_coords(ro.parse_smiles("CCO"))
        c = m.coords_3d

        def dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

        # bond 0-1 (C-C) and 1-2 (C-O)
        for bond_idx in range(m.num_bonds):
            bond = m.get_bond(bond_idx)
            d = dist(c[bond.source_idx], c[bond.target_idx])
            assert 0.5 <= d <= 4.0, \
                f"Unreasonable bond length {d:.2f} Å for bond {bond_idx}"

    def test_no_two_atoms_same_3d_position(self):
        """No two atoms should be at exactly the same 3D position."""
        m = ro.generate_3d_coords(ro.parse_smiles("C1CCCCC1"))
        c = m.coords_3d
        n = len(c)

        def dist_sq(a, b):
            return sum((a[i] - b[i]) ** 2 for i in range(3))

        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq(c[i], c[j])
                assert d2 > 1e-6, \
                    f"Atoms {i} and {j} at same position: {c[i]}"


# ─── AMW ordering invariants ─────────────────────────────────────────────────

class TestAmwOrdering:

    def test_methane_lighter_than_ethane(self):
        assert ro.parse_smiles("C").amw < ro.parse_smiles("CC").amw

    def test_benzene_lighter_than_naphthalene(self):
        assert ro.parse_smiles("c1ccccc1").amw < ro.parse_smiles("c1ccc2ccccc2c1").amw

    def test_adding_heavy_atom_increases_amw(self):
        """Adding a bromine (heavy) should increase AMW more than adding C."""
        base  = ro.parse_smiles("CC")          # ethane ~24 (heavy atoms only)
        plus_c = ro.parse_smiles("CCC")        # + one C
        plus_br = ro.parse_smiles("CCBr")      # + one Br
        delta_c  = plus_c.amw  - base.amw
        delta_br = plus_br.amw - base.amw
        assert delta_br > delta_c  # Br (80) >> C (12)



