use crate::molecule::RustMolecule;
use crate::atom::Atom;
use crate::bond::BondType;
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a SMILES string into a RustMolecule.
///
/// Returns `Err` on:
///   - Unknown element in organic subset or bracket atom
///   - Unclosed `[` bracket
///   - Unclosed ring closure  (e.g. `C1CC`)
///   - Unclosed branch        (e.g. `CC(CC`)
///   - Empty or whitespace-only input
pub fn parse_smiles(smiles: &str) -> Result<RustMolecule, String> {
    let trimmed = smiles.trim();
    if trimmed.is_empty() {
        return Err("SMILES parse error: empty string".to_string());
    }

    let mut mol = RustMolecule::new();
    let mut atom_stack: Vec<usize> = Vec::new();
    // ring_closures: ring_num -> (opening_atom_idx, explicit_bond_at_opening)
    let mut ring_closures: HashMap<u16, (usize, Option<BondType>)> = HashMap::new();
    let mut last_atom_idx: Option<usize> = None;
    let mut chars = trimmed.chars().peekable();
    // None = implicit bond (Single or Aromatic based on atom context)
    let mut current_bond: Option<BondType> = None;

    while let Some(&c) = chars.peek() {
        match c {
            '(' => {
                chars.next();
                match last_atom_idx {
                    Some(idx) => { atom_stack.push(idx); current_bond = None; }
                    None => return Err("SMILES parse error: branch '(' without preceding atom".to_string()),
                }
            }
            ')' => {
                chars.next();
                match atom_stack.pop() {
                    Some(idx) => { last_atom_idx = Some(idx); current_bond = None; }
                    None => return Err("SMILES parse error: unbalanced branch ')'".to_string()),
                }
            }
            '-' | '=' | '#' | ':' => {
                chars.next();
                current_bond = Some(match c {
                    '-' => BondType::Single,
                    '=' => BondType::Double,
                    '#' => BondType::Triple,
                    ':' => BondType::Aromatic,
                    _ => unreachable!(),
                });
            }
            '/' | '\\' => {
                chars.next();
                current_bond = Some(BondType::Single); // stereo → treat as single
            }
            '[' => {
                chars.next();
                let mut bracket_content = String::new();
                let mut closed = false;
                while let Some(&bc) = chars.peek() {
                    if bc == ']' { chars.next(); closed = true; break; }
                    bracket_content.push(chars.next().unwrap());
                }
                if !closed {
                    return Err("SMILES parse error: unclosed '[' bracket".to_string());
                }
                let (atom, _override) = parse_bracket_atom(&bracket_content)?;
                let is_aromatic = atom.is_aromatic;
                let atom_idx = mol.num_atoms();
                mol.add_atom(atom);
                if let Some(prev) = last_atom_idx {
                    let prev_arom = mol.get_atom(prev).map_or(false, |a| a.is_aromatic);
                    let bt = implicit_or_explicit(current_bond, prev_arom, is_aromatic);
                    mol.add_bond(prev, atom_idx, bt);
                }
                last_atom_idx = Some(atom_idx);
                current_bond = None;
            }
            '0'..='9' | '%' => {
                let ring_num: u16 = if c == '%' {
                    chars.next(); // consume '%'
                    let mut s = String::new();
                    // Read exactly 2 digits (SMILES spec: %dd)
                    while s.len() < 2 {
                        match chars.peek() {
                            Some(&nc) if nc.is_ascii_digit() => { s.push(chars.next().unwrap()); }
                            _ => break,
                        }
                    }
                    if s.is_empty() {
                        return Err("SMILES parse error: invalid ring number after '%'".to_string());
                    }
                    s.parse().map_err(|_| "SMILES parse error: invalid ring number".to_string())?
                } else {
                    chars.next().unwrap().to_digit(10).unwrap() as u16
                };

                let curr_atom = match last_atom_idx {
                    Some(idx) => idx,
                    None => return Err("SMILES parse error: ring closure digit without preceding atom".to_string()),
                };

                match ring_closures.remove(&ring_num) {
                    Some((partner, opening_bond)) => {
                        // Close the ring
                        let is_curr  = mol.get_atom(curr_atom).map_or(false, |a| a.is_aromatic);
                        let is_part  = mol.get_atom(partner).map_or(false, |a| a.is_aromatic);
                        let bt = current_bond.or(opening_bond).unwrap_or_else(|| {
                            if is_curr && is_part { BondType::Aromatic } else { BondType::Single }
                        });
                        mol.add_bond(curr_atom, partner, bt);
                    }
                    None => {
                        // Open the ring
                        ring_closures.insert(ring_num, (curr_atom, current_bond));
                    }
                }
                current_bond = None;
            }
            '.' => {
                // Disconnected fragment separator
                chars.next();
                last_atom_idx = None;
                current_bond = None;
            }
            _ => {
                let first = chars.next().unwrap();
                let mut symbol = first.to_string();
                // Two-char organic-subset elements: Cl, Br
                if (first == 'C' && chars.peek() == Some(&'l'))
                    || (first == 'B' && chars.peek() == Some(&'r'))
                {
                    symbol.push(chars.next().unwrap());
                }
                let atomic_number = match symbol.as_str() {
                    "C" | "c" => 6,  "N" | "n" => 7,  "O" | "o" => 8,
                    "F"        => 9,  "P" | "p" => 15, "S" | "s" => 16,
                    "Cl"       => 17, "Br"      => 35, "I" | "i" => 53,
                    "H"        => 1,  "B" | "b" => 5,
                    other => return Err(format!("SMILES parse error: unknown element '{}'", other)),
                };
                let is_aromatic = first.is_lowercase();
                let mut atom = Atom::new(atomic_number, 0, 0, is_aromatic);
                atom.implicit_valence = match atomic_number {
                    6 => 4, 7 => 3, 8 => 2, 16 => 2,
                    9 | 17 | 35 | 53 => 1, _ => 0,
                };
                let atom_idx = mol.num_atoms();
                mol.add_atom(atom);
                if let Some(prev) = last_atom_idx {
                    let prev_arom = mol.get_atom(prev).map_or(false, |a| a.is_aromatic);
                    let bt = implicit_or_explicit(current_bond, prev_arom, is_aromatic);
                    mol.add_bond(prev, atom_idx, bt);
                }
                last_atom_idx = Some(atom_idx);
                current_bond = None;
            }
        }
    }

    // ── post-parse validation ──────────────────────────────────────────────
    if !atom_stack.is_empty() {
        return Err(format!("SMILES parse error: {} unclosed branch(es) '('", atom_stack.len()));
    }
    if !ring_closures.is_empty() {
        let mut keys: Vec<u16> = ring_closures.keys().cloned().collect();
        keys.sort_unstable();
        return Err(format!("SMILES parse error: unclosed ring closure(s): {:?}", keys));
    }

    Ok(mol)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond-type helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn implicit_or_explicit(explicit: Option<BondType>, prev_arom: bool, curr_arom: bool) -> BondType {
    match explicit {
        Some(bt) => bt,
        None => if prev_arom && curr_arom { BondType::Aromatic } else { BondType::Single },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bracket-atom parser
// ─────────────────────────────────────────────────────────────────────────────

fn parse_bracket_atom(content: &str) -> Result<(Atom, Option<BondType>), String> {
    let mut chars = content.chars().peekable();

    // Optional isotope digits
    while chars.peek().map_or(false, |c| c.is_ascii_digit()) { chars.next(); }

    // Element symbol
    let symbol = read_element_symbol(&mut chars)?;
    let atomic_number = element_to_atomic_number(&symbol)
        .ok_or_else(|| format!("SMILES parse error: unknown element '[{}]'", symbol))?;
    let is_aromatic = symbol.chars().next().map_or(false, |c| c.is_lowercase());

    // Skip stereochemistry descriptors (@ or @@)
    while chars.peek() == Some(&'@') { chars.next(); }

    // Explicit H count: H or Hn
    let mut num_hs = 0u8;
    if chars.peek() == Some(&'H') {
        chars.next();
        match chars.peek() {
            Some(&c) if c.is_ascii_digit() => {
                num_hs = chars.next().unwrap().to_digit(10).unwrap() as u8;
            }
            _ => num_hs = 1,
        }
    }

    // Formal charge: +, -, +n, -n, ++, --
    let mut charge = 0i8;
    loop {
        match chars.peek().copied() {
            Some('+') => {
                chars.next();
                if chars.peek().map_or(false, |c| c.is_ascii_digit()) {
                    charge += chars.next().unwrap().to_digit(10).unwrap() as i8;
                } else {
                    charge += 1;
                }
            }
            Some('-') => {
                chars.next();
                if chars.peek().map_or(false, |c| c.is_ascii_digit()) {
                    charge -= chars.next().unwrap().to_digit(10).unwrap() as i8;
                } else {
                    charge -= 1;
                }
            }
            _ => break,
        }
    }

    Ok((Atom::new(atomic_number, charge, num_hs, is_aromatic), None))
}

fn read_element_symbol(
    chars: &mut std::iter::Peekable<std::str::Chars>,
) -> Result<String, String> {
    let first = chars.next().ok_or("SMILES parse error: empty bracket atom")?;
    let mut sym = first.to_string();

    // Possibly read one lowercase letter to form a 2-char symbol (e.g. Na, Mg, Fe)
    if first.is_uppercase() {
        if let Some(&second) = chars.peek() {
            if second.is_lowercase() {
                // Try 2-char first
                let candidate = format!("{}{}", first, second);
                if element_to_atomic_number(&candidate).is_some() {
                    sym = candidate;
                    chars.next();
                }
                // else: single-char symbol, leave the lowercase for H/charge parsing
            }
        }
    }
    Ok(sym)
}

fn element_to_atomic_number(sym: &str) -> Option<u16> {
    match sym {
        // Organic subset + aromatic lower-case
        "H"  | "h"  => Some(1),
        "B"  | "b"  => Some(5),
        "C"  | "c"  => Some(6),
        "N"  | "n"  => Some(7),
        "O"  | "o"  => Some(8),
        "F"  | "f"  => Some(9),
        "P"  | "p"  => Some(15),
        "S"  | "s"  => Some(16),
        "I"  | "i"  => Some(53),
        // Two-char (always title-case in SMILES)
        "He" => Some(2),  "Li" => Some(3),  "Be" => Some(4),
        "Ne" => Some(10), "Na" => Some(11), "Mg" => Some(12),
        "Al" => Some(13), "Si" => Some(14), "Cl" => Some(17),
        "Ar" => Some(18), "K"  => Some(19), "Ca" => Some(20),
        "Sc" => Some(21), "Ti" => Some(22), "V"  => Some(23),
        "Cr" => Some(24), "Mn" => Some(25), "Fe" => Some(26),
        "Co" => Some(27), "Ni" => Some(28), "Cu" => Some(29),
        "Zn" => Some(30), "Ga" => Some(31), "Ge" => Some(32),
        "As" | "as" => Some(33),
        "Se" | "se" => Some(34),
        "Br" => Some(35), "Kr" => Some(36), "Rb" => Some(37),
        "Sr" => Some(38), "Y"  => Some(39), "Zr" => Some(40),
        "Nb" => Some(41), "Mo" => Some(42), "Tc" => Some(43),
        "Ru" => Some(44), "Rh" => Some(45), "Pd" => Some(46),
        "Ag" => Some(47), "Cd" => Some(48), "In" => Some(49),
        "Sn" => Some(50), "Sb" => Some(51),
        "Te" | "te" => Some(52),
        "Xe" => Some(54), "Cs" => Some(55), "Ba" => Some(56),
        "La" => Some(57), "Hf" => Some(72), "Ta" => Some(73),
        "W"  => Some(74), "Re" => Some(75), "Os" => Some(76),
        "Ir" => Some(77), "Pt" => Some(78), "Au" => Some(79),
        "Hg" => Some(80), "Tl" => Some(81), "Pb" => Some(82),
        "Bi" => Some(83),
        "*"  => Some(0), // wildcard atom
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical SMILES
// ─────────────────────────────────────────────────────────────────────────────

pub fn canonicalize(mol: &RustMolecule) -> String {
    let n = mol.num_atoms();
    if n == 0 { return String::new(); }

    let rank = compute_canonical_ranks(mol);
    // Start from the atom that has rank 0 (lowest canonical priority)
    let start = (0..n).min_by_key(|&i| rank[i]).unwrap_or(0);

    // ── Pre-pass DFS: classify edges into tree edges vs ring bonds ──────────
    let mut ring_bonds: Vec<(usize, usize, BondType)> = Vec::new();
    {
        let mut visited = HashSet::new();
        let mut seen_rb  = HashSet::new();
        find_ring_bonds_dfs(start, None, mol, &rank, &mut visited, &mut ring_bonds, &mut seen_rb);
    }

    // Set of ring-bond atom pairs (canonical pair: min < max)
    let ring_bond_set: HashSet<(usize, usize)> = ring_bonds.iter()
        .map(|&(u, v, _)| (u.min(v), u.max(v)))
        .collect();

    // Assign ring numbers; record at both endpoints
    let mut ring_closures_at: HashMap<usize, Vec<(usize, BondType)>> = HashMap::new();
    for (idx, &(u, v, bt)) in ring_bonds.iter().enumerate() {
        let rnum = idx + 1;
        ring_closures_at.entry(u).or_default().push((rnum, bt));
        ring_closures_at.entry(v).or_default().push((rnum, bt));
    }
    // Sort by ring number for determinism
    for v in ring_closures_at.values_mut() { v.sort_by_key(|&(r, _)| r); }

    // ── Emit canonical SMILES ───────────────────────────────────────────────
    let mut smiles = String::new();
    let mut visited = HashSet::new();
    emit_canonical(start, mol, &rank, &ring_bond_set, &ring_closures_at, &mut visited, &mut smiles);
    smiles
}

// ── Morgan-based canonical rank computation ──────────────────────────────────

fn compute_canonical_ranks(mol: &RustMolecule) -> Vec<usize> {
    let n = mol.num_atoms();

    // Initial per-atom hashes
    let mut hashes: Vec<u64> = (0..n).map(|i| {
        let a = &mol.inner.atoms[i];
        let deg = mol.inner.bonds.iter()
            .filter(|b| b.source_idx == i || b.target_idx == i)
            .count() as u64;
        let mut h = fnv_init();
        h = fnv(h, a.atomic_number as u64);
        h = fnv(h, deg);
        h = fnv(h, (a.formal_charge as i64 + 64) as u64);
        h = fnv(h, a.num_explicit_hs as u64);
        h = fnv(h, a.is_aromatic as u64);
        h
    }).collect();

    // 5 Morgan rounds
    for _ in 0..5 {
        let prev = hashes.clone();
        for i in 0..n {
            let mut nbr_hashes: Vec<u64> = mol.inner.bonds.iter()
                .filter(|b| b.source_idx == i || b.target_idx == i)
                .map(|b| {
                    let j = if b.source_idx == i { b.target_idx } else { b.source_idx };
                    prev[j]
                })
                .collect();
            nbr_hashes.sort_unstable();
            let mut h = prev[i];
            for nh in &nbr_hashes { h = fnv(h, *nh); }
            hashes[i] = h;
        }
    }

    // Assign rank: sort by (hash, original_idx) for tie-breaking
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| hashes[a].cmp(&hashes[b]).then(a.cmp(&b)));

    let mut rank = vec![0usize; n];
    for (r, &atom) in order.iter().enumerate() { rank[atom] = r; }
    rank
}

// ── Pre-pass DFS to find ring (back-edge) bonds ───────────────────────────────

fn find_ring_bonds_dfs(
    curr: usize,
    parent: Option<usize>,
    mol: &RustMolecule,
    rank: &[usize],
    visited: &mut HashSet<usize>,
    ring_bonds: &mut Vec<(usize, usize, BondType)>,
    seen_rb: &mut HashSet<(usize, usize)>,
) {
    visited.insert(curr);

    let mut neighbors: Vec<(usize, BondType)> = mol.inner.bonds.iter()
        .filter(|b| b.source_idx == curr || b.target_idx == curr)
        .map(|b| {
            let n = if b.source_idx == curr { b.target_idx } else { b.source_idx };
            (n, b.bond_type)
        })
        .collect();
    // Sort by canonical rank so ring bond assignment is deterministic
    neighbors.sort_by_key(|&(n, _)| rank[n]);

    for (n, bt) in neighbors {
        if parent == Some(n) { continue; } // skip the edge we came from
        if visited.contains(&n) {
            // Back-edge → ring bond (record only once)
            let key = (curr.min(n), curr.max(n));
            if !seen_rb.contains(&key) {
                seen_rb.insert(key);
                ring_bonds.push((curr, n, bt));
            }
        } else {
            find_ring_bonds_dfs(n, Some(curr), mol, rank, visited, ring_bonds, seen_rb);
        }
    }
}

// ── Canonical SMILES emission ─────────────────────────────────────────────────

fn emit_canonical(
    curr: usize,
    mol: &RustMolecule,
    rank: &[usize],
    ring_bond_set: &HashSet<(usize, usize)>,
    ring_closures_at: &HashMap<usize, Vec<(usize, BondType)>>,
    visited: &mut HashSet<usize>,
    smiles: &mut String,
) {
    visited.insert(curr);
    let atom = mol.get_atom(curr).unwrap();

    // ── Atom symbol ──────────────────────────────────────────────────────────
    let sym_base = &atom.symbol;
    let sym = if atom.is_aromatic { sym_base.to_lowercase() } else { sym_base.clone() };

    if atom.formal_charge != 0 || atom.num_explicit_hs > 0 {
        smiles.push('[');
        smiles.push_str(&sym);
        match atom.num_explicit_hs {
            0 => {}
            1 => smiles.push('H'),
            n => { smiles.push('H'); smiles.push_str(&n.to_string()); }
        }
        match atom.formal_charge.cmp(&0) {
            std::cmp::Ordering::Greater => { smiles.push('+'); smiles.push_str(&atom.formal_charge.to_string()); }
            std::cmp::Ordering::Less    => smiles.push_str(&atom.formal_charge.to_string()),
            std::cmp::Ordering::Equal   => {}
        }
        smiles.push(']');
    } else {
        smiles.push_str(&sym);
    }

    // ── Ring closure digits ──────────────────────────────────────────────────
    if let Some(closures) = ring_closures_at.get(&curr) {
        for &(rnum, bt) in closures {
            // Emit bond char only for non-default bond types in ring closures
            match bt {
                BondType::Double  => smiles.push('='),
                BondType::Triple  => smiles.push('#'),
                _ => {} // Single and Aromatic are implicit
            }
            push_ring_num(smiles, rnum);
        }
    }

    // ── Tree-edge children ───────────────────────────────────────────────────
    // Only recurse into neighbors that are (a) unvisited AND (b) not ring bonds
    let mut children: Vec<(usize, BondType)> = mol.inner.bonds.iter()
        .filter(|b| b.source_idx == curr || b.target_idx == curr)
        .map(|b| {
            let n = if b.source_idx == curr { b.target_idx } else { b.source_idx };
            (n, b.bond_type)
        })
        .filter(|(n, _)| {
            !visited.contains(n)
            && !ring_bond_set.contains(&(curr.min(*n), curr.max(*n)))
        })
        .collect();
    // Sort children by canonical rank (ascending = lower-rank first)
    children.sort_by_key(|&(n, _)| rank[n]);

    let len = children.len();
    for (k, (n, bt)) in children.into_iter().enumerate() {
        let b_char: &str = match bt {
            BondType::Double  => "=",
            BondType::Triple  => "#",
            _ => "", // Single/Aromatic are implicit
        };
        if k < len - 1 {
            smiles.push('(');
            smiles.push_str(b_char);
            emit_canonical(n, mol, rank, ring_bond_set, ring_closures_at, visited, smiles);
            smiles.push(')');
        } else {
            smiles.push_str(b_char);
            emit_canonical(n, mol, rank, ring_bond_set, ring_closures_at, visited, smiles);
        }
    }
}

#[inline]
fn push_ring_num(smiles: &mut String, rnum: usize) {
    if rnum >= 10 {
        smiles.push_str(&format!("%{:02}", rnum));
    } else {
        smiles.push(char::from_digit(rnum as u32, 10).unwrap());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline] fn fnv_init() -> u64 { 2166136261u64 }
#[inline] fn fnv(h: u64, v: u64) -> u64 { h.wrapping_mul(16777619).wrapping_add(v) }
