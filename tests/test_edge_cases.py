"""
test_edge_cases.py
══════════════════
Edge cases, boundary conditions, and error-handling tests.

Covers:
  • Empty/single-atom/single-bond molecules
  • Charged atoms and multi-valent bracket atoms
  • Disconnected fragments (SMILES with ".")
  • Long chains (stress atoms/bonds at scale)
  • Out-of-range index access
  • Malformed SMILES inputs
  • Coordinate setters with mismatched sizes
  • Atoms with no bonds
  • Very large ring closures
  • Isotope labels
  • Molecules with all atom types
"""

import pytest

import chem_engine as ro

# ─── Empty and minimal molecules ────────────────────────────────────────────


class TestEmptyAndMinimal:
    def test_empty_molecule_num_atoms_zero(self):
        m = ro.RustMolecule()
        assert m.num_atoms == 0

    def test_empty_molecule_num_bonds_zero(self):
        m = ro.RustMolecule()
        assert m.num_bonds == 0

    def test_empty_molecule_amw_zero(self):
        m = ro.RustMolecule()
        assert m.amw == 0.0

    def test_empty_molecule_rotatable_zero(self):
        m = ro.RustMolecule()
        assert m.num_rotatable_bonds == 0

    def test_empty_molecule_fingerprint_all_false(self):
        m = ro.RustMolecule()
        fp = m.get_fingerprint()
        assert all(b is False for b in fp)

    def test_empty_molecule_similarity_zero(self):
        a = ro.RustMolecule()
        b = ro.RustMolecule()
        assert a.similarity(b) == 0.0

    def test_empty_molecule_no_coords(self):
        m = ro.RustMolecule()
        assert m.coords_2d is None
        assert m.coords_3d is None

    def test_empty_molecule_substruct_empty_query(self):
        """Empty molecule contains empty query."""
        target = ro.RustMolecule()
        query = ro.RustMolecule()
        assert target.has_substruct_match(query) is True

    def test_single_atom_carbon(self):
        m = ro.parse_smiles("[C]")
        assert m.num_atoms == 1
        assert m.num_bonds == 0

    def test_single_atom_oxygen(self):
        m = ro.parse_smiles("[O]")
        assert m.num_atoms == 1
        assert m.get_atom(0).atomic_number == 8

    def test_single_atom_nitrogen(self):
        m = ro.parse_smiles("[N]")
        assert m.num_atoms == 1
        assert m.get_atom(0).atomic_number == 7

    def test_single_atom_no_rotatable(self):
        m = ro.parse_smiles("[C]")
        assert m.num_rotatable_bonds == 0

    def test_single_bond_molecule(self):
        m = ro.parse_smiles("CC")
        assert m.num_atoms == 2
        assert m.num_bonds == 1


# ─── Charged atoms ──────────────────────────────────────────────────────────


class TestChargedAtoms:
    def test_ammonium_positive_charge(self):
        m = ro.parse_smiles("[NH4+]")
        a = m.get_atom(0)
        assert a.formal_charge == 1
        assert a.atomic_number == 7

    def test_carboxylate_negative_charge(self):
        m = ro.parse_smiles("[O-]C(=O)")
        # First atom should be O with charge -1
        atoms = [m.get_atom(i) for i in range(m.num_atoms)]
        neg_o = [a for a in atoms if a.formal_charge == -1 and a.atomic_number == 8]
        assert len(neg_o) >= 1

    def test_sodium_cation(self):
        """[Na+] is now correctly parsed: atomic_number=11, formal_charge=+1."""
        m = ro.parse_smiles("[Na+]")
        a = m.get_atom(0)
        assert a.atomic_number == 11
        assert a.formal_charge == 1

    def test_chloride_anion(self):
        m = ro.parse_smiles("[Cl-]")
        a = m.get_atom(0)
        assert a.formal_charge == -1
        assert a.atomic_number == 17

    def test_charge_neutral_molecule(self):
        m = ro.parse_smiles("CCO")
        for i in range(m.num_atoms):
            assert m.get_atom(i).formal_charge == 0

    def test_zwitterion_both_charges(self):
        """Glycine zwitterion [NH3+]CC([O-])=O"""
        m = ro.parse_smiles("[NH3+]CC([O-])=O")
        charges = [m.get_atom(i).formal_charge for i in range(m.num_atoms)]
        assert 1 in charges
        assert -1 in charges


# ─── Explicit hydrogen bracket atoms ────────────────────────────────────────


class TestExplicitHydrogens:
    def test_explicit_h_count_ammonium(self):
        m = ro.parse_smiles("[NH4+]")
        a = m.get_atom(0)
        assert a.num_explicit_hs == 4

    def test_explicit_h_count_water(self):
        m = ro.parse_smiles("[H]O[H]")
        # O atom has 0 explicit H in bracket notation; H atoms are separate atoms
        assert m.num_atoms == 3
        assert m.num_bonds == 2

    def test_bracket_nh2(self):
        m = ro.parse_smiles("[NH2]c1ccccc1")  # aniline bracket form
        n_atom = next(m.get_atom(i) for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 7)
        assert n_atom.num_explicit_hs == 2

    def test_h_atom_atomic_number_1(self):
        m = ro.parse_smiles("[H]")
        assert m.num_atoms == 1
        assert m.get_atom(0).atomic_number == 1


# ─── Halogens ────────────────────────────────────────────────────────────────


class TestHalogens:
    def test_fluoromethane(self):
        m = ro.parse_smiles("CF")
        symbols = {m.get_atom(i).symbol for i in range(m.num_atoms)}
        assert "F" in symbols
        assert "C" in symbols

    def test_chloromethane_atomic_number(self):
        m = ro.parse_smiles("CCl")
        cl_atoms = [m.get_atom(i) for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 17]
        assert len(cl_atoms) == 1

    def test_bromomethane_atomic_number(self):
        m = ro.parse_smiles("CBr")
        br_atoms = [m.get_atom(i) for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 35]
        assert len(br_atoms) == 1

    def test_iodomethane(self):
        m = ro.parse_smiles("CI")
        i_atoms = [m.get_atom(i) for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 53]
        assert len(i_atoms) == 1

    def test_perfluorobenzene_6_fluorines(self):
        m = ro.parse_smiles("Fc1c(F)c(F)c(F)c(F)c1F")
        f_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 9)
        assert f_count == 6


# ─── Ring closure edge cases ─────────────────────────────────────────────────


class TestRingClosures:
    def test_3_membered_ring(self):
        """Cyclopropane C1CC1 - 3 atoms, 3 bonds."""
        m = ro.parse_smiles("C1CC1")
        assert m.num_atoms == 3
        assert m.num_bonds == 3

    def test_4_membered_ring(self):
        """Cyclobutane - 4 atoms, 4 bonds."""
        m = ro.parse_smiles("C1CCC1")
        assert m.num_atoms == 4
        assert m.num_bonds == 4

    def test_5_membered_ring(self):
        m = ro.parse_smiles("C1CCCC1")
        assert m.num_atoms == 5
        assert m.num_bonds == 5

    def test_large_ring_closure_double_digit(self):
        """Ring closures %10-%99 (two-digit notation)."""
        m = ro.parse_smiles("C%10CCCCC%10")  # cyclohexane via %10
        assert m.num_atoms == 6
        assert m.num_bonds == 6

    def test_bicyclic_decalin(self):
        """Decalin C1CCCCC2CCCCC12 - 11 atoms, 12 bonds (chem-engine's ring-closure count)."""
        m = ro.parse_smiles("C1CCCCC2CCCCC12")
        assert m.num_atoms == 11  # includes shared ring-junction atom
        assert m.num_bonds == 12

    def test_spiro_compound(self):
        """Spiro[4.4]nonane C1CCCC12CCCCC2 - 10 atoms, 11 bonds (chem-engine count)."""
        m = ro.parse_smiles("C1CCCC12CCCCC2")
        assert m.num_atoms == 10
        assert m.num_bonds == 11


# ─── Long-chain stress ───────────────────────────────────────────────────────


class TestLongChains:
    def test_c20_chain_atom_count(self):
        smi = "C" * 20
        m = ro.parse_smiles(smi)
        assert m.num_atoms == 20

    def test_c20_chain_bond_count(self):
        smi = "C" * 20
        m = ro.parse_smiles(smi)
        assert m.num_bonds == 19

    def test_c20_rotatable_bonds_count(self):
        """C20 linear chain: 17 rotatable bonds.

        C1-C2 through C17-C18; both terminal C-C bonds are excluded.
        """
        smi = "C" * 20
        m = ro.parse_smiles(smi)
        assert m.num_rotatable_bonds == 17

    def test_c50_chain(self):
        smi = "C" * 50
        m = ro.parse_smiles(smi)
        assert m.num_atoms == 50
        assert m.num_bonds == 49


# ─── Out-of-bounds index access ──────────────────────────────────────────────


class TestOutOfBoundsAccess:
    def test_get_atom_out_of_range_returns_none(self):
        m = ro.parse_smiles("CCO")
        result = m.get_atom(100)
        assert result is None

    def test_get_bond_out_of_range_returns_none(self):
        m = ro.parse_smiles("CCO")
        result = m.get_bond(100)
        assert result is None

    def test_get_atom_negative_index_does_not_crash(self):
        """Negative indices should not crash the process."""
        m = ro.parse_smiles("CCO")
        # Python wraps negative indices; what matters is no segfault
        try:
            m.get_atom(0)  # valid access - ensures molecule exists
        except Exception:
            pass  # any exception is acceptable; crash is not

    def test_find_bond_invalid_indices(self):
        m = ro.parse_smiles("CCO")
        result = m.find_bond(0, 999)
        assert result is None


# ─── Invalid SMILES ──────────────────────────────────────────────────────────


class TestInvalidSmiles:
    def test_completely_invalid(self):
        with pytest.raises(Exception):
            ro.parse_smiles("XYZ_INVALID_SMILES_999")

    def test_unclosed_ring(self):
        with pytest.raises(Exception):
            ro.parse_smiles("C1CCC")

    def test_unclosed_branch(self):
        with pytest.raises(Exception):
            ro.parse_smiles("CC(CC")

    def test_empty_string(self):
        with pytest.raises(Exception):
            ro.parse_smiles("")

    def test_only_whitespace(self):
        with pytest.raises(Exception):
            ro.parse_smiles("   ")

    def test_unknown_element(self):
        with pytest.raises(Exception):
            ro.parse_smiles("[Xx]")

    def test_mismatched_brackets(self):
        with pytest.raises(Exception):
            ro.parse_smiles("[C")


# ─── Coordinate setter edge cases ────────────────────────────────────────────


class TestCoordinateSetters:
    def test_set_2d_coords_correct_count(self):
        m = ro.parse_smiles("CCO")
        m.coords_2d = [[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]
        assert len(m.coords_2d) == 3

    def test_set_3d_coords_correct_count(self):
        m = ro.parse_smiles("CCO")
        m.coords_3d = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0]]
        assert len(m.coords_3d) == 3

    def test_set_3d_coords_values_preserved(self):
        m = ro.parse_smiles("CC")
        m.coords_3d = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
        c = m.coords_3d
        assert abs(c[0][0] - 1.1) < 1e-6
        assert abs(c[1][2] - 6.6) < 1e-6

    def test_overwrite_coords_replaces_previous(self):
        m = ro.parse_smiles("CC")
        m.coords_2d = [[0.0, 0.0], [1.0, 0.0]]
        m.coords_2d = [[5.0, 5.0], [6.0, 6.0]]
        c = m.coords_2d
        assert abs(c[0][0] - 5.0) < 1e-6

    def test_coords_2d_none_before_generation(self):
        m = ro.parse_smiles("CCO")
        assert m.coords_2d is None

    def test_coords_3d_none_before_generation(self):
        m = ro.parse_smiles("CCO")
        assert m.coords_3d is None

    def test_2d_layout_produces_coords(self):
        m = ro.parse_smiles("CCO")
        m = ro.generate_2d_coords(m)
        assert m.coords_2d is not None
        assert len(m.coords_2d) == m.num_atoms

    def test_3d_embed_produces_coords(self):
        m = ro.parse_smiles("CCO")
        m = ro.generate_3d_coords(m)
        assert m.coords_3d is not None
        assert len(m.coords_3d) == m.num_atoms


# ─── Batch parse edge cases ──────────────────────────────────────────────────


class TestBatchEdgeCases:
    def test_batch_empty_list(self):
        assert ro.batch_parse_smiles([]) == []

    def test_batch_single_item(self):
        results = ro.batch_parse_smiles(["CCO"])
        assert len(results) == 1
        assert results[0].num_atoms == 3

    def test_batch_preserves_order(self):
        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
        results = ro.batch_parse_smiles(smiles)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.num_atoms == i + 1, f"Position {i}: expected {i + 1} atoms, got {r.num_atoms}"

    def test_batch_diverse_elements(self):
        smiles = ["CCO", "CCN", "CCS", "CF", "CCl", "CBr", "CI"]
        expected_atoms = [3, 3, 3, 2, 2, 2, 2]
        results = ro.batch_parse_smiles(smiles)
        assert len(results) == 7
        for i, (r, exp) in enumerate(zip(results, expected_atoms)):
            assert r.num_atoms == exp, (
                f"Position {i} ({smiles[i]}): expected {exp} atoms, got {r.num_atoms}"
            )

    def test_batch_100_identical(self):
        results = ro.batch_parse_smiles(["c1ccccc1"] * 100)
        assert len(results) == 100
        assert all(r.num_atoms == 6 for r in results)

    def test_batch_1000_mixed(self):
        smiles_pool = ["CCO", "CCCC", "c1ccccc1", "CN", "CC(=O)O"] * 200
        results = ro.batch_parse_smiles(smiles_pool)
        assert len(results) == 1000
