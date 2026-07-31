use pyo3::prelude::*;
use crate::atom::Atom;
use crate::bond::{Bond, BondType};
use std::sync::Arc;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct MoleculeData {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub coords_2d: Option<Vec<[f64; 2]>>,
    pub coords_3d: Option<Vec<[f64; 3]>>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct RustMolecule {
    pub inner: Arc<MoleculeData>,
}

unsafe impl Send for RustMolecule {}
unsafe impl Sync for RustMolecule {}

#[pymethods]
impl RustMolecule {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MoleculeData {
                atoms: Vec::new(),
                bonds: Vec::new(),
                coords_2d: None,
                coords_3d: None,
            }),
        }
    }

    pub fn add_atom(&mut self, atom: Atom) {
        let mut data = (*self.inner).clone();
        data.atoms.push(atom);
        self.inner = Arc::new(data);
    }

    pub fn add_bond(&mut self, source_idx: usize, target_idx: usize, bond_type: BondType) {
        let mut data = (*self.inner).clone();
        data.bonds.push(Bond::new(source_idx, target_idx, bond_type));
        self.inner = Arc::new(data);
    }

    #[getter]
    pub fn num_atoms(&self) -> usize {
        self.inner.atoms.len()
    }

    #[getter]
    pub fn num_bonds(&self) -> usize {
        self.inner.bonds.len()
    }

    pub fn get_atom(&self, idx: usize) -> Option<Atom> {
        self.inner.atoms.get(idx).cloned()
    }

    pub fn get_bond(&self, idx: usize) -> Option<Bond> {
        self.inner.bonds.get(idx).cloned()
    }

    #[getter]
    pub fn amw(&self) -> f64 {
        let atom_mass: f64 = self.inner.atoms.iter().map(|a| a.get_mass()).sum();
        let implicit_hs_mass: f64 = self.inner.atoms.iter().map(|a| a.num_explicit_hs as f64 * 1.008).sum();
        atom_mass + implicit_hs_mass
    }

    #[getter]
    pub fn num_rotatable_bonds(&self) -> usize {
        let num_atoms = self.inner.atoms.len();
        if num_atoms == 0 {
            return 0;
        }

        let mut rotatable_count = 0;
        for (i, bond) in self.inner.bonds.iter().enumerate() {
            // Only single bonds can be rotatable
            if bond.bond_type != BondType::Single {
                continue;
            }

            let u = bond.source_idx;
            let v = bond.target_idx;

            if u >= num_atoms || v >= num_atoms {
                continue;
            }

            // Terminal atoms (degree 1) make a bond non-rotatable
            let deg_u = self.inner.bonds.iter().filter(|b| b.source_idx == u || b.target_idx == u).count();
            let deg_v = self.inner.bonds.iter().filter(|b| b.source_idx == v || b.target_idx == v).count();

            if deg_u <= 1 || deg_v <= 1 {
                continue;
            }

            // Ring bonds (non-bridges) are not rotatable.
            // is_bridge() returns true when removing the bond disconnects the graph,
            // meaning the bond IS a bridge (acyclic). Acyclic non-terminal single bonds
            // are rotatable, so we count when is_bridge() == true.
            if self.is_bridge(u, v, i) {
                rotatable_count += 1;
            }
        }

        rotatable_count
    }

    pub fn find_bond(&self, u: usize, v: usize) -> Option<Bond> {
        self.inner.bonds.iter().find(|b| {
            (b.source_idx == u && b.target_idx == v) || (b.source_idx == v && b.target_idx == u)
        }).cloned()
    }

    // FR-9: Substructure Search
    pub fn has_substruct_match(&self, query: &RustMolecule) -> bool {
        let q_atoms = &query.inner.atoms;
        let t_atoms = &self.inner.atoms;

        if q_atoms.is_empty() {
            return true;
        }
        if t_atoms.len() < q_atoms.len() {
            return false;
        }

        let mut assignment = vec![None; q_atoms.len()];
        let mut assigned_targets = HashSet::new();

        self.backtrack_match(0, &mut assignment, &mut assigned_targets, query)
    }

    // FR-10: Similarity Search — ECFP2-style Morgan fingerprint (radius 2)
    pub fn get_fingerprint(&self) -> Vec<bool> {
        let mut fp = vec![false; 2048];
        let n = self.inner.atoms.len();
        if n == 0 { return fp; }

        // Pre-compute atom degrees
        let degrees: Vec<u64> = (0..n).map(|i| {
            self.inner.bonds.iter()
                .filter(|b| b.source_idx == i || b.target_idx == i)
                .count() as u64
        }).collect();

        // Round 0: per-atom identifier = FNV hash of (atomic_num, degree, charge, hs, aromatic)
        let mut env: Vec<u64> = (0..n).map(|i| {
            let a = &self.inner.atoms[i];
            let mut h: u64 = 2166136261;
            h = h.wrapping_mul(16777619).wrapping_add(a.atomic_number as u64);
            h = h.wrapping_mul(16777619).wrapping_add(degrees[i]);
            h = h.wrapping_mul(16777619).wrapping_add((a.formal_charge as i64 + 64) as u64);
            h = h.wrapping_mul(16777619).wrapping_add(a.num_explicit_hs as u64);
            h = h.wrapping_mul(16777619).wrapping_add(a.is_aromatic as u64);
            fp[(h % 2048) as usize] = true;
            h
        }).collect();

        // Rounds 1 and 2: Morgan-style neighbourhood aggregation
        for _ in 0..2 {
            let prev = env.clone();
            for i in 0..n {
                // Collect (bond_type, neighbour_hash) pairs, sorted for invariance
                let mut nbr: Vec<(u64, u64)> = self.inner.bonds.iter()
                    .filter(|b| b.source_idx == i || b.target_idx == i)
                    .map(|b| {
                        let j = if b.source_idx == i { b.target_idx } else { b.source_idx };
                        (b.bond_type as u64, prev[j])
                    })
                    .collect();
                nbr.sort_unstable();
                let mut h: u64 = 2166136261;
                h = h.wrapping_mul(16777619).wrapping_add(prev[i]);
                for (bt, nh) in &nbr {
                    h = h.wrapping_mul(16777619).wrapping_add(*bt);
                    h = h.wrapping_mul(16777619).wrapping_add(*nh);
                }
                env[i] = h;
                fp[(h % 2048) as usize] = true;
            }
        }

        fp
    }

    pub fn similarity(&self, other: &RustMolecule) -> f64 {
        let fp1 = self.get_fingerprint();
        let fp2 = other.get_fingerprint();
        
        let mut intersection = 0;
        let mut union = 0;
        for i in 0..2048 {
            if fp1[i] && fp2[i] {
                intersection += 1;
            }
            if fp1[i] || fp2[i] {
                union += 1;
            }
        }
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    // FR-7: Tautomer Enumeration and Standardization
    pub fn enumerate_tautomers(&self) -> Vec<RustMolecule> {
        let mut tautomers = vec![self.clone()];
        let num_atoms = self.inner.atoms.len();
        
        // Rules engine to find keto-enol / amide-imidic tautomeric systems
        // Pattern: [O,N,S]=[C,N]-[C,N]-[H] (1-3 proton shift)
        for bond_idx in 0..self.inner.bonds.len() {
            let bond = &self.inner.bonds[bond_idx];
            if bond.bond_type != BondType::Double {
                continue;
            }

            let u = bond.source_idx;
            let v = bond.target_idx;
            if u >= num_atoms || v >= num_atoms {
                continue;
            }

            let atom_u = &self.inner.atoms[u];
            let atom_v = &self.inner.atoms[v];

            // Identify double bond center: one heteroatom (O, N, S) and one carbon/nitrogen
            let (hetero_idx, carbon_idx) = if (atom_u.atomic_number == 8 || atom_u.atomic_number == 7 || atom_u.atomic_number == 16)
                && (atom_v.atomic_number == 6 || atom_v.atomic_number == 7) {
                (u, v)
            } else if (atom_v.atomic_number == 8 || atom_v.atomic_number == 7 || atom_v.atomic_number == 16)
                && (atom_u.atomic_number == 6 || atom_u.atomic_number == 7) {
                (v, u)
            } else {
                continue;
            };

            // Look for neighbor of carbon_idx connected via a single bond
            for neighbor_bond in &self.inner.bonds {
                if neighbor_bond.bond_type != BondType::Single {
                    continue;
                }
                let c_neigh = if neighbor_bond.source_idx == carbon_idx {
                    neighbor_bond.target_idx
                } else if neighbor_bond.target_idx == carbon_idx {
                    neighbor_bond.source_idx
                } else {
                    continue;
                };

                if c_neigh == hetero_idx || c_neigh >= num_atoms {
                    continue;
                }

                let atom_neigh = &self.inner.atoms[c_neigh];
                // Check if neighbor has shiftable hydrogen (explicit or implicit)
                if atom_neigh.num_explicit_hs > 0 || atom_neigh.implicit_valence > 0 {
                    // Create tautomer molecule by shifting proton
                    let mut t_mol = self.clone();
                    let mut t_data = (*t_mol.inner).clone();
                    
                    // 1. Shift H from neighbor to heteroatom
                    if t_data.atoms[c_neigh].num_explicit_hs > 0 {
                        t_data.atoms[c_neigh].num_explicit_hs -= 1;
                    }
                    t_data.atoms[hetero_idx].num_explicit_hs += 1;

                    // 2. Adjust bond orders: double bond becomes single, single becomes double
                    for b in &mut t_data.bonds {
                        if (b.source_idx == hetero_idx && b.target_idx == carbon_idx) || (b.source_idx == carbon_idx && b.target_idx == hetero_idx) {
                            b.bond_type = BondType::Single;
                        }
                        if (b.source_idx == carbon_idx && b.target_idx == c_neigh) || (b.source_idx == c_neigh && b.target_idx == carbon_idx) {
                            b.bond_type = BondType::Double;
                        }
                    }

                    t_mol.inner = Arc::new(t_data);
                    tautomers.push(t_mol);
                }
            }
        }

        tautomers
    }

    pub fn get_canonical_tautomer(&self) -> RustMolecule {
        let tautomers = self.enumerate_tautomers();
        let mut best_tautomer = self.clone();
        let mut best_score = -1000;

        for t in tautomers {
            let score = score_tautomer(&t);
            if score > best_score {
                best_score = score;
                best_tautomer = t;
            }
        }

        best_tautomer
    }

    #[getter]
    pub fn coords_2d(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coords_2d.as_ref().map(|coords| {
            coords.iter().map(|c| vec![c[0], c[1]]).collect()
        })
    }

    #[setter]
    pub fn set_coords_2d(&mut self, coords: Vec<Vec<f64>>) {
        let mut data = (*self.inner).clone();
        let mut c_2d = Vec::new();
        for c in coords {
            if c.len() >= 2 {
                c_2d.push([c[0], c[1]]);
            }
        }
        data.coords_2d = Some(c_2d);
        self.inner = Arc::new(data);
    }

    #[getter]
    pub fn coords_3d(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.coords_3d.as_ref().map(|coords| {
            coords.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
        })
    }

    #[setter]
    pub fn set_coords_3d(&mut self, coords: Vec<Vec<f64>>) {
        let mut data = (*self.inner).clone();
        let mut c_3d = Vec::new();
        for c in coords {
            if c.len() >= 3 {
                c_3d.push([c[0], c[1], c[2]]);
            }
        }
        data.coords_3d = Some(c_3d);
        self.inner = Arc::new(data);
    }
}

impl RustMolecule {
    fn is_bridge(&self, u: usize, v: usize, exclude_bond_idx: usize) -> bool {
        let num_atoms = self.inner.atoms.len();
        let mut visited = vec![false; num_atoms];
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back(u);
        visited[u] = true;

        while let Some(curr) = queue.pop_front() {
            if curr == v {
                return false;
            }
            for (idx, bond) in self.inner.bonds.iter().enumerate() {
                if idx == exclude_bond_idx {
                    continue;
                }
                let neighbor = if bond.source_idx == curr {
                    Some(bond.target_idx)
                } else if bond.target_idx == curr {
                    Some(bond.source_idx)
                } else {
                    None
                };

                if let Some(n) = neighbor {
                    if n < num_atoms && !visited[n] {
                        visited[n] = true;
                        queue.push_back(n);
                    }
                }
            }
        }

        true
    }

    fn backtrack_match(
        &self,
        q_idx: usize,
        assignment: &mut [Option<usize>],
        assigned_targets: &mut HashSet<usize>,
        query: &RustMolecule,
    ) -> bool {
        if q_idx == query.inner.atoms.len() {
            return true;
        }

        let q_atom = &query.inner.atoms[q_idx];
        let t_atoms = &self.inner.atoms;

        for t_idx in 0..t_atoms.len() {
            if assigned_targets.contains(&t_idx) {
                continue;
            }

            let t_atom = &t_atoms[t_idx];

            if q_atom.atomic_number != t_atom.atomic_number || q_atom.is_aromatic != t_atom.is_aromatic {
                continue;
            }

            let mut bonds_ok = true;
            for q_prev in 0..q_idx {
                if let Some(t_prev) = assignment[q_prev] {
                    if let Some(q_bond) = query.find_bond(q_idx, q_prev) {
                        if let Some(t_bond) = self.find_bond(t_idx, t_prev) {
                            if q_bond.bond_type != t_bond.bond_type {
                                bonds_ok = false;
                                break;
                            }
                        } else {
                            bonds_ok = false;
                            break;
                        }
                    }
                }
            }

            if !bonds_ok {
                continue;
            }

            assignment[q_idx] = Some(t_idx);
            assigned_targets.insert(t_idx);

            if self.backtrack_match(q_idx + 1, assignment, assigned_targets, query) {
                return true;
            }

            assignment[q_idx] = None;
            assigned_targets.remove(&t_idx);
        }

        false
    }
}

fn score_tautomer(mol: &RustMolecule) -> i32 {
    let mut score = 0;
    
    // Keto form preferenced over enol form
    // Count double bonded oxygen C=O
    for bond in &mol.inner.bonds {
        if bond.bond_type == BondType::Double {
            let u_atom = &mol.inner.atoms[bond.source_idx];
            let v_atom = &mol.inner.atoms[bond.target_idx];
            
            if (u_atom.atomic_number == 8 && v_atom.atomic_number == 6) ||
               (v_atom.atomic_number == 8 && u_atom.atomic_number == 6) {
                score += 15;
            }

            // Amide C(=O)NH preferences
            if (u_atom.atomic_number == 7 && v_atom.atomic_number == 6) ||
               (v_atom.atomic_number == 7 && u_atom.atomic_number == 6) {
                score += 5;
            }
        }
    }

    // Aromaticity score: reward aromatic elements
    for atom in &mol.inner.atoms {
        if atom.is_aromatic {
            score += 10;
        }
    }

    score
}
