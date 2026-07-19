use crate::molecule::RustMolecule;
use crate::atom::Atom;
use crate::bond::{Bond, BondType};
use std::collections::{HashMap, HashSet};

pub fn parse_smiles(smiles: &str) -> Result<RustMolecule, String> {
    let mut mol = RustMolecule::new();
    let mut atom_stack: Vec<usize> = Vec::new();
    let mut ring_closures: HashMap<u16, usize> = HashMap::new();
    let mut last_atom_idx: Option<usize> = None;
    let mut chars = smiles.chars().peekable();
    let mut current_bond = BondType::Single;

    while let Some(&c) = chars.peek() {
        match c {
            '(' => {
                chars.next();
                if let Some(idx) = last_atom_idx {
                    atom_stack.push(idx);
                } else {
                    return Err("SMILES parse error: branch '(' without preceding atom".to_string());
                }
            }
            ')' => {
                chars.next();
                if let Some(idx) = atom_stack.pop() {
                    last_atom_idx = Some(idx);
                } else {
                    return Err("SMILES parse error: unbalanced branch ')'".to_string());
                }
            }
            '-' | '=' | '#' | ':' => {
                chars.next();
                current_bond = match c {
                    '-' => BondType::Single,
                    '=' => BondType::Double,
                    '#' => BondType::Triple,
                    ':' => BondType::Aromatic,
                    _ => BondType::Single,
                };
            }
            '/' | '\\' => {
                chars.next(); // stereochemistry bonds, treat as single for layout/graphs
                current_bond = BondType::Single;
            }
            '[' => {
                chars.next();
                // Read properties inside brackets, e.g. [nH], [O-], [CH3]
                let mut bracket_content = String::new();
                while let Some(&bc) = chars.peek() {
                    if bc == ']' {
                        chars.next();
                        break;
                    }
                    bracket_content.push(chars.next().unwrap());
                }
                
                let (atom, bond_override) = parse_bracket_atom(&bracket_content)?;
                let atom_idx = mol.num_atoms();
                mol.add_atom(atom);

                if let Some(prev) = last_atom_idx {
                    let final_bond = if bond_override.is_some() { bond_override.unwrap() } else { current_bond };
                    mol.add_bond(prev, atom_idx, final_bond);
                }

                last_atom_idx = Some(atom_idx);
                current_bond = BondType::Single;
            }
            '0'..='9' | '%' => {
                let ring_num = if c == '%' {
                    chars.next(); // skip '%'
                    let mut num_str = String::new();
                    while let Some(&nc) = chars.peek() {
                        if nc.is_digit(10) {
                            num_str.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    num_str.parse::<u16>().map_err(|_| "SMILES parse error: invalid ring number after %")?
                } else {
                    chars.next().unwrap().to_digit(10).unwrap() as u16
                };

                if let Some(idx) = last_atom_idx {
                    if let Some(&partner_idx) = ring_closures.get(&ring_num) {
                        mol.add_bond(idx, partner_idx, current_bond);
                        ring_closures.remove(&ring_num);
                    } else {
                        ring_closures.insert(ring_num, idx);
                    }
                } else {
                    return Err("SMILES parse error: ring closure digit without preceding atom".to_string());
                }
                current_bond = BondType::Single;
            }
            _ => {
                // Parse standard atom (C, N, O, etc.)
                let symbol_char = chars.next().unwrap();
                let mut symbol = symbol_char.to_string();
                
                // Check for two-character elements Cl, Br
                if (symbol_char == 'C' && chars.peek() == Some(&'l')) || 
                   (symbol_char == 'B' && chars.peek() == Some(&'r')) {
                    symbol.push(chars.next().unwrap());
                }

                let atomic_number = match symbol.as_str() {
                    "C" | "c" => 6,
                    "N" | "n" => 7,
                    "O" | "o" => 8,
                    "F" | "f" => 9,
                    "P" | "p" => 15,
                    "S" | "s" => 16,
                    "Cl" => 17,
                    "Br" => 35,
                    "I" | "i" => 53,
                    "H" => 1,
                    "B" | "b" => 5,
                    _ => return Err(format!("SMILES parse error: unknown element '{}'", symbol)),
                };

                let is_aromatic = symbol.chars().next().unwrap().is_lowercase();
                
                let mut atom = Atom::new(atomic_number, 0, 0, is_aromatic);
                // Compute implicit hydrogens based on valence
                atom.implicit_valence = match atomic_number {
                    6 => 4,
                    7 => 3,
                    8 => 2,
                    16 => 2,
                    9 | 17 | 35 | 53 => 1,
                    _ => 0,
                };
                
                let atom_idx = mol.num_atoms();
                mol.add_atom(atom);

                if let Some(prev) = last_atom_idx {
                    mol.add_bond(prev, atom_idx, current_bond);
                }

                last_atom_idx = Some(atom_idx);
                current_bond = BondType::Single;
            }
        }
    }

    Ok(mol)
}

fn parse_bracket_atom(content: &str) -> Result<(Atom, Option<BondType>), String> {
    // E.g., "nH", "O-", "CH3", "13C", "C@@H"
    let mut chars = content.chars().peekable();
    
    // Parse isotope if present
    let mut isotope = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_digit(10) {
            isotope.push(chars.next().unwrap());
        } else {
            break;
        }
    }

    // Parse atomic symbol
    let mut symbol = String::new();
    if let Some(c) = chars.next() {
        symbol.push(c);
        if c == 'C' && chars.peek() == Some(&'l') {
            symbol.push(chars.next().unwrap());
        } else if c == 'B' && chars.peek() == Some(&'r') {
            symbol.push(chars.next().unwrap());
        }
    } else {
        return Err("SMILES parse error: empty bracket atom".to_string());
    }

    let atomic_number = match symbol.as_str() {
        "C" | "c" => 6,
        "N" | "n" => 7,
        "O" | "o" => 8,
        "F" | "f" => 9,
        "P" | "p" => 15,
        "S" | "s" => 16,
        "Cl" => 17,
        "Br" => 35,
        "I" | "i" => 53,
        "H" => 1,
        "B" | "b" => 5,
        _ => 6, // default to carbon if parsing fails
    };

    let is_aromatic = symbol.chars().next().unwrap().is_lowercase();

    // Parse stereochemical indicators (like @, @@)
    while let Some(&c) = chars.peek() {
        if c == '@' {
            chars.next();
        } else {
            break;
        }
    }

    // Parse explicit hydrogens
    let mut num_hs = 0;
    if chars.peek() == Some(&'H') {
        chars.next();
        if let Some(&c) = chars.peek() {
            if c.is_digit(10) {
                num_hs = chars.next().unwrap().to_digit(10).unwrap() as u8;
            } else {
                num_hs = 1;
            }
        } else {
            num_hs = 1;
        }
    }

    // Parse formal charge, e.g., "+", "-", "+2", "-1"
    let mut charge = 0i8;
    if let Some(c) = chars.next() {
        if c == '+' || c == '-' {
            let sign = if c == '+' { 1 } else { -1 };
            let mut val_str = String::new();
            while let Some(&nc) = chars.peek() {
                if nc.is_digit(10) {
                    val_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            let val = if val_str.is_empty() { 1 } else { val_str.parse::<i8>().unwrap_or(1) };
            charge = sign * val;
        }
    }

    let atom = Atom::new(atomic_number, charge, num_hs, is_aromatic);
    Ok((atom, None))
}

pub fn canonicalize(mol: &RustMolecule) -> String {
    let num_atoms = mol.num_atoms();
    if num_atoms == 0 {
        return "".to_string();
    }

    // 1. Compute node invariants: (atomic_number, degree, formal_charge, num_hs, atom_idx)
    let mut invariants: Vec<(u16, usize, i8, u8, usize)> = (0..num_atoms).map(|i| {
        let atom = mol.get_atom(i).unwrap();
        let degree = mol.inner.bonds.iter().filter(|b| b.source_idx == i || b.target_idx == i).count();
        (atom.atomic_number, degree, atom.formal_charge, atom.num_explicit_hs, i)
    }).collect();

    // 2. Morgan-like refinement of invariants (5 rounds)
    for _ in 0..5 {
        let mut next_invariants = invariants.clone();
        for i in 0..num_atoms {
            let neighbors: Vec<usize> = mol.inner.bonds.iter()
                .filter(|b| b.source_idx == i || b.target_idx == i)
                .map(|b| if b.source_idx == i { b.target_idx } else { b.source_idx })
                .collect();
            let mut neighbor_invs: Vec<u16> = neighbors.iter().map(|&n| invariants[n].0).collect();
            neighbor_invs.sort();
            let n_sum: u16 = neighbor_invs.iter().sum();
            next_invariants[i].0 = (next_invariants[i].0 as u32 + n_sum as u32) as u16;
        }
        invariants = next_invariants;
    }

    // Sort atoms by refined invariant (lowest invariant = highest priority start)
    invariants.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
            .then(a.4.cmp(&b.4))
    });

    // Build canonical rank for each atom index
    let mut rank = vec![0usize; num_atoms];
    for (r, inv) in invariants.iter().enumerate() {
        rank[inv.4] = r;
    }

    // 3. DFS traversal from lowest-ranked atom, emitting ring closure digits
    let start_atom = invariants[0].4;
    let mut visited = HashSet::new();
    let mut ring_closures: HashMap<(usize, usize), usize> = HashMap::new(); // edge -> ring_num
    let mut ring_counter = 1usize;
    let mut smiles = String::new();

    dfs_canonical(
        start_atom,
        mol,
        &rank,
        &mut visited,
        &mut ring_closures,
        &mut ring_counter,
        &mut smiles,
    );

    smiles
}

fn dfs_canonical(
    curr: usize,
    mol: &RustMolecule,
    rank: &[usize],
    visited: &mut HashSet<usize>,
    ring_closures: &mut HashMap<(usize, usize), usize>,
    ring_counter: &mut usize,
    smiles: &mut String,
) {
    visited.insert(curr);
    let atom = mol.get_atom(curr).unwrap();

    // Emit atom symbol
    let mut sym = atom.symbol.clone();
    if atom.is_aromatic {
        sym = sym.to_lowercase();
    }

    if atom.formal_charge != 0 || atom.num_explicit_hs > 0 {
        smiles.push('[');
        smiles.push_str(&sym);
        if atom.num_explicit_hs > 0 {
            if atom.num_explicit_hs == 1 {
                smiles.push('H');
            } else {
                smiles.push_str(&format!("H{}", atom.num_explicit_hs));
            }
        }
        if atom.formal_charge > 0 {
            smiles.push_str(&format!("+{}", atom.formal_charge));
        } else if atom.formal_charge < 0 {
            smiles.push_str(&format!("{}", atom.formal_charge));
        }
        smiles.push(']');
    } else {
        smiles.push_str(&sym);
    }

    // Collect neighbors sorted by canonical rank
    let mut neighbors: Vec<(usize, BondType, usize)> = mol.inner.bonds.iter()
        .filter(|b| b.source_idx == curr || b.target_idx == curr)
        .map(|b| {
            let n = if b.source_idx == curr { b.target_idx } else { b.source_idx };
            (rank[n], b.bond_type, n)
        })
        .collect();
    neighbors.sort_by_key(|n| n.0);

    // Emit ring closure digits for already-visited back-edges
    for &(_, bond_type, neighbor_idx) in &neighbors {
        if visited.contains(&neighbor_idx) {
            let edge_key = if curr < neighbor_idx { (curr, neighbor_idx) } else { (neighbor_idx, curr) };
            if !ring_closures.contains_key(&edge_key) {
                let rnum = *ring_counter;
                *ring_counter += 1;
                ring_closures.insert(edge_key, rnum);
                // Emit bond char and ring number
                let b_char = match bond_type {
                    BondType::Single => "",
                    BondType::Double => "=",
                    BondType::Triple => "#",
                    BondType::Aromatic => ":",
                };
                smiles.push_str(b_char);
                smiles.push_str(&rnum.to_string());
            }
        }
    }

    // DFS into unvisited neighbors, using branches where needed
    let unvisited: Vec<(usize, BondType, usize)> = neighbors.iter()
        .filter(|(_, _, n)| !visited.contains(n))
        .cloned()
        .collect();

    for (k, &(_, bond_type, neighbor_idx)) in unvisited.iter().enumerate() {
        let b_char = match bond_type {
            BondType::Single => "",
            BondType::Double => "=",
            BondType::Triple => "#",
            BondType::Aromatic => ":",
        };

        if k < unvisited.len() - 1 {
            // All but the last neighbor need branch parentheses
            smiles.push('(');
            smiles.push_str(b_char);
            dfs_canonical(neighbor_idx, mol, rank, visited, ring_closures, ring_counter, smiles);
            smiles.push(')');
        } else {
            // Last (or only) neighbor: inline continuation
            smiles.push_str(b_char);
            dfs_canonical(neighbor_idx, mol, rank, visited, ring_closures, ring_counter, smiles);
        }
    }
}

