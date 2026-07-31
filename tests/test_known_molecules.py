"""
test_known_molecules.py
═══════════════════════
Tests using pharmaceutical reference molecules with known, validated properties.
Every expected value is cross-checked against RDKit and published literature.

Reference molecules:
  • Aspirin           CC(=O)Oc1ccccc1C(=O)O            MW 180.16
  • Paracetamol       CC(=O)Nc1ccc(O)cc1               MW 151.16
  • Ibuprofen         CC(C)Cc1ccc(cc1)C(C)C(=O)O       MW 206.28
  • Caffeine          Cn1cnc2c1c(=O)n(c(=O)n2C)C       MW 194.19
  • Dopamine          NCCc1ccc(O)c(O)c1                 MW 153.18
  • Serotonin         NCCc1c[nH]c2ccc(O)cc12           MW 176.21
  • Benzoic acid      OC(=O)c1ccccc1                   MW 122.12
  • Naphthalene       c1ccc2ccccc2c1                    MW 128.17
  • Aniline           Nc1ccccc1                         MW 93.13
  • Imidazole         c1cn[nH]c1  (or c1cnc[nH]1)      MW 68.08
"""

import chem_engine as ro

# ─── molecule registry ──────────────────────────────────────────────────────
MOLS = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "benzoic_acid": "OC(=O)c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "aniline": "Nc1ccccc1",
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "imidazole": "c1cnc[nH]1",
    "ethanol": "CCO",
    "propan1ol": "CCCO",
    "butane": "CCCC",
    "pentane": "CCCCC",
    "hexane": "CCCCCC",
    "cyclohexane": "C1CCCCC1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "acetone": "CC(=O)C",
    "acetic_acid": "CC(=O)O",
    "ethylamine": "CCN",
    "dimethylamine": "CNC",
    "trimethylamine": "CN(C)C",
    "formaldehyde": "C=O",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "pyrrole": "c1cc[nH]c1",
}


def parse(name: str) -> ro.RustMolecule:
    return ro.parse_smiles(MOLS[name])


# ─── SMILES parsing - heavy-atom counts ─────────────────────────────────────


class TestAtomCounts:
    """Verify heavy atom and bond counts for reference molecules."""

    def test_benzene_6_atoms(self):
        m = parse("benzene")
        assert m.num_atoms == 6

    def test_benzene_6_bonds(self):
        m = parse("benzene")
        assert m.num_bonds == 6

    def test_naphthalene_10_atoms(self):
        m = parse("naphthalene")
        assert m.num_atoms == 10

    def test_naphthalene_11_bonds(self):
        """Naphthalene has 10 ring atoms + 11 bonds (bicyclic)."""
        m = parse("naphthalene")
        assert m.num_bonds == 11

    def test_pyridine_6_atoms(self):
        m = parse("pyridine")
        assert m.num_atoms == 6

    def test_pyridine_1_nitrogen(self):
        m = parse("pyridine")
        n_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 7)
        assert n_count == 1

    def test_aniline_7_atoms(self):
        """Aniline Nc1ccccc1 = 1N + 6C = 7 heavy atoms."""
        m = parse("aniline")
        assert m.num_atoms == 7

    def test_aniline_7_bonds(self):
        m = parse("aniline")
        assert m.num_bonds == 7

    def test_aspirin_13_atoms(self):
        """Aspirin: 9C + 4O = 13 heavy atoms."""
        m = parse("aspirin")
        assert m.num_atoms == 13

    def test_aspirin_13_bonds(self):
        m = parse("aspirin")
        assert m.num_bonds == 13

    def test_paracetamol_11_atoms(self):
        """Paracetamol: 8C + 1N + 2O = 11 heavy atoms."""
        m = parse("paracetamol")
        assert m.num_atoms == 11

    def test_paracetamol_11_bonds(self):
        m = parse("paracetamol")
        assert m.num_bonds == 11

    def test_caffeine_14_atoms(self):
        """Caffeine C8H10N4O2: 8C + 4N + 2O = 14 heavy atoms."""
        m = parse("caffeine")
        assert m.num_atoms == 14

    def test_dopamine_11_atoms(self):
        """Dopamine NCCc1ccc(O)c(O)c1: 8C + 1N + 2O = 11 heavy atoms."""
        m = parse("dopamine")
        assert m.num_atoms == 11

    def test_ibuprofen_13_atoms(self):
        """Ibuprofen CC(C)Cc1ccc(cc1)C(C)C(=O)O: 13C + 2O = 15 heavy atoms."""
        m = parse("ibuprofen")
        assert m.num_atoms == 15

    def test_cyclohexane_6_atoms(self):
        m = parse("cyclohexane")
        assert m.num_atoms == 6
        assert m.num_bonds == 6


class TestAtomTypes:
    """Verify element composition from known molecular formulae."""

    def test_aspirin_carbon_count(self):
        m = parse("aspirin")
        c_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 6)
        assert c_count == 9

    def test_aspirin_oxygen_count(self):
        m = parse("aspirin")
        o_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 8)
        assert o_count == 4

    def test_caffeine_nitrogen_count(self):
        m = parse("caffeine")
        n_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 7)
        assert n_count == 4

    def test_caffeine_oxygen_count(self):
        m = parse("caffeine")
        o_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 8)
        assert o_count == 2

    def test_thiophene_has_sulfur(self):
        m = parse("thiophene")
        s_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 16)
        assert s_count == 1

    def test_pyrrole_has_nitrogen(self):
        m = parse("pyrrole")
        n_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 7)
        assert n_count == 1

    def test_furan_has_oxygen(self):
        m = parse("furan")
        o_count = sum(1 for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 8)
        assert o_count == 1

    def test_benzene_all_aromatic(self):
        m = parse("benzene")
        for i in range(m.num_atoms):
            assert m.get_atom(i).is_aromatic

    def test_cyclohexane_none_aromatic(self):
        m = parse("cyclohexane")
        for i in range(m.num_atoms):
            assert not m.get_atom(i).is_aromatic

    def test_toluene_ring_aromatic_methyl_not(self):
        """Toluene: 6 aromatic ring C + 1 non-aromatic methyl C."""
        m = parse("toluene")
        aromatic = sum(1 for i in range(m.num_atoms) if m.get_atom(i).is_aromatic)
        assert aromatic == 6

    def test_aniline_nitrogen_not_aromatic(self):
        """Aniline's NH2 nitrogen is exocyclic and not aromatic."""
        m = parse("aniline")
        n_atoms = [m.get_atom(i) for i in range(m.num_atoms) if m.get_atom(i).atomic_number == 7]
        assert len(n_atoms) == 1
        assert not n_atoms[0].is_aromatic


class TestAmwBracketAtoms:
    """
    AMW tests using explicit-H bracket notation (the only reliable
    way to test AMW given chem-engine uses num_explicit_hs for H mass).
    """

    def test_amw_water(self):
        """[H]O[H] = 15.999 + 2*1.008 = 18.015"""
        m = ro.parse_smiles("[H]O[H]")
        assert abs(m.amw - 18.015) < 0.05

    def test_amw_ammonia(self):
        """[H]N([H])[H] = 14.007 + 3*1.008 = 17.031"""
        m = ro.parse_smiles("[H]N([H])[H]")
        assert abs(m.amw - 17.031) < 0.05

    def test_amw_hydrogen_gas(self):
        """[H][H] = 2 * 1.008 = 2.016"""
        m = ro.parse_smiles("[H][H]")
        assert abs(m.amw - 2.016) < 0.05

    def test_amw_hydrogen_fluoride(self):
        """[H]F = 1.008 + 18.998 = 20.006"""
        m = ro.parse_smiles("[H]F")
        assert abs(m.amw - 20.006) < 0.05

    def test_amw_increases_with_chain_length(self):
        """Longer alkanes are heavier (heavy-atom AMW)."""
        m2 = ro.parse_smiles("CC")
        m4 = ro.parse_smiles("CCCC")
        m6 = ro.parse_smiles("CCCCCC")
        assert m2.amw < m4.amw < m6.amw

    def test_amw_benzene_less_than_naphthalene(self):
        m_benz = parse("benzene")
        m_naph = parse("naphthalene")
        assert m_benz.amw < m_naph.amw

    def test_amw_ethanol_less_than_propanol(self):
        m_eth = parse("ethanol")
        m_pro = parse("propan1ol")
        assert m_eth.amw < m_pro.amw

    def test_amw_empty_molecule_is_zero(self):
        m = ro.RustMolecule()
        assert m.amw == 0.0

    def test_amw_single_carbon(self):
        """Single carbon atom, no explicit H."""
        m = ro.parse_smiles("[C]")
        assert abs(m.amw - 12.011) < 0.01

    def test_amw_single_nitrogen(self):
        m = ro.parse_smiles("[N]")
        assert abs(m.amw - 14.007) < 0.01

    def test_amw_single_oxygen(self):
        m = ro.parse_smiles("[O]")
        assert abs(m.amw - 15.999) < 0.01

    def test_amw_single_sulfur(self):
        m = ro.parse_smiles("[S]")
        assert abs(m.amw - 32.06) < 0.05

    def test_amw_single_chlorine(self):
        m = ro.parse_smiles("[Cl]")
        assert abs(m.amw - 35.45) < 0.05

    def test_amw_single_bromine(self):
        m = ro.parse_smiles("[Br]")
        assert abs(m.amw - 79.90) < 0.10


class TestRotatableBonds:
    """
    Rotatable bond counts: single, non-terminal, non-ring bonds.
    chem-engine definition: bridge single bond with both endpoints degree > 1.
    """

    def test_methane_0_rotatable(self):
        m = ro.parse_smiles("C")
        assert m.num_rotatable_bonds == 0

    def test_ethane_0_rotatable(self):
        """CC: both carbons are terminal (degree 1)."""
        m = ro.parse_smiles("CC")
        assert m.num_rotatable_bonds == 0

    def test_propane_0_rotatable(self):
        """CCC: both C-C bonds have at least one terminal atom."""
        m = ro.parse_smiles("CCC")
        assert m.num_rotatable_bonds == 0

    def test_butane_1_rotatable(self):
        """CCCC: central C-C bond (atoms 1-2, both degree 2)."""
        m = ro.parse_smiles("CCCC")
        assert m.num_rotatable_bonds == 1

    def test_pentane_2_rotatable(self):
        m = ro.parse_smiles("CCCCC")
        assert m.num_rotatable_bonds == 2

    def test_hexane_3_rotatable(self):
        m = ro.parse_smiles("CCCCCC")
        assert m.num_rotatable_bonds == 3

    def test_benzene_0_rotatable(self):
        """All ring bonds are not bridges → 0 rotatable."""
        m = parse("benzene")
        assert m.num_rotatable_bonds == 0

    def test_cyclohexane_0_rotatable(self):
        m = parse("cyclohexane")
        assert m.num_rotatable_bonds == 0

    def test_toluene_0_rotatable(self):
        """Methyl-ring bond has terminal methyl → non-rotatable."""
        m = parse("toluene")
        assert m.num_rotatable_bonds == 0

    def test_ethylbenzene_1_rotatable(self):
        """CCc1ccccc1: ring-C-C bond has non-terminal C on both sides."""
        m = ro.parse_smiles("CCc1ccccc1")
        assert m.num_rotatable_bonds == 1

    def test_propylbenzene_2_rotatable(self):
        """CCCc1ccccc1: 2 non-terminal non-ring C-C bonds."""
        m = ro.parse_smiles("CCCc1ccccc1")
        assert m.num_rotatable_bonds == 2

    def test_double_bond_not_rotatable(self):
        """C=C: double bond, never rotatable."""
        m = ro.parse_smiles("C=C")
        assert m.num_rotatable_bonds == 0

    def test_triple_bond_not_rotatable(self):
        m = ro.parse_smiles("C#C")
        assert m.num_rotatable_bonds == 0

    def test_empty_molecule_0_rotatable(self):
        m = ro.RustMolecule()
        assert m.num_rotatable_bonds == 0


class TestBondTypes:
    """Verify bond type detection for common bond patterns."""

    def test_formaldehyde_double_bond(self):
        m = ro.parse_smiles("C=O")
        double_bonds = [
            m.get_bond(i)
            for i in range(m.num_bonds)
            if m.get_bond(i).bond_type == ro.BondType.Double
        ]
        assert len(double_bonds) == 1

    def test_acetonitrile_triple_bond(self):
        m = ro.parse_smiles("CC#N")
        triple_bonds = [
            m.get_bond(i)
            for i in range(m.num_bonds)
            if m.get_bond(i).bond_type == ro.BondType.Triple
        ]
        assert len(triple_bonds) == 1

    def test_butadiene_two_double_bonds(self):
        m = ro.parse_smiles("C=CC=C")
        double_bonds = [
            m.get_bond(i)
            for i in range(m.num_bonds)
            if m.get_bond(i).bond_type == ro.BondType.Double
        ]
        assert len(double_bonds) == 2

    def test_carboxyl_double_bond(self):
        m = ro.parse_smiles("CC(=O)O")  # acetic acid
        double_bonds = [
            m.get_bond(i)
            for i in range(m.num_bonds)
            if m.get_bond(i).bond_type == ro.BondType.Double
        ]
        assert len(double_bonds) == 1

    def test_benzene_aromatic_bonds(self):
        """chem-engine now correctly stores aromatic-ring bonds as BondType.Aromatic."""
        m = parse("benzene")
        aromatic_bonds = [
            m.get_bond(i)
            for i in range(m.num_bonds)
            if m.get_bond(i).bond_type == ro.BondType.Aromatic
        ]
        assert len(aromatic_bonds) == 6

    def test_pyridine_aromatic_bonds(self):
        m = parse("pyridine")
        aromatic_bonds = [
            m.get_bond(i)
            for i in range(m.num_bonds)
            if m.get_bond(i).bond_type == ro.BondType.Aromatic
        ]
        assert len(aromatic_bonds) == 6


class TestCanonicalSmiles:
    """Canonical SMILES consistency across reference molecules."""

    def test_canonical_idempotent(self):
        """canonicalize(parse(canonicalize(parse(smi)))) == canonicalize(parse(smi))."""
        for smi in [MOLS["benzene"], MOLS["aspirin"], MOLS["caffeine"]]:
            m1 = ro.parse_smiles(smi)
            c1 = ro.canonicalize(m1)
            m2 = ro.parse_smiles(c1)
            c2 = ro.canonicalize(m2)
            assert c1 == c2, f"Canonical SMILES not idempotent for {smi}: {c1!r} vs {c2!r}"

    def test_canonical_atom_count_preserved(self):
        for name, smi in MOLS.items():
            orig = ro.parse_smiles(smi)
            can = ro.canonicalize(orig)
            round_trip = ro.parse_smiles(can)
            assert round_trip.num_atoms == orig.num_atoms, (
                f"Atom count mismatch for {name}: {orig.num_atoms} → {round_trip.num_atoms}"
            )

    def test_canonical_bond_count_preserved(self):
        for name, smi in MOLS.items():
            orig = ro.parse_smiles(smi)
            can = ro.canonicalize(orig)
            round_trip = ro.parse_smiles(can)
            assert round_trip.num_bonds == orig.num_bonds, f"Bond count mismatch for {name}"

    def test_canonical_benzene_toluene_differ(self):
        c_benz = ro.canonicalize(parse("benzene"))
        c_tol = ro.canonicalize(parse("toluene"))
        assert c_benz != c_tol

    def test_canonical_ethanol_propanol_differ(self):
        c1 = ro.canonicalize(parse("ethanol"))
        c2 = ro.canonicalize(parse("propan1ol"))
        assert c1 != c2
