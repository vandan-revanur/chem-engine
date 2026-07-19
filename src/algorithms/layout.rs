use crate::molecule::RustMolecule;
use crate::bond::BondType;
use std::sync::Arc;

pub fn generate_2d_coords(mol: &mut RustMolecule) {
    let num_atoms = mol.num_atoms();
    if num_atoms == 0 {
        return;
    }

    // 1. Initialize coordinates in a circle to avoid overlapping starts
    let mut coords = vec![[0.0, 0.0]; num_atoms];
    for i in 0..num_atoms {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_atoms as f64);
        coords[i] = [angle.cos() * 2.0, angle.sin() * 2.0];
    }

    // 2. Force-directed layout iterations
    let iterations = 100;
    let k_spring = 0.15;
    let k_repulsion = 0.5;
    let d_zero = 1.5; // Ideal bond length
    let dt = 0.1;

    for _ in 0..iterations {
        let mut forces = vec![[0.0, 0.0]; num_atoms];

        // Repulsion forces (all pairs)
        for i in 0..num_atoms {
            for j in (i + 1)..num_atoms {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dist_sq = dx * dx + dy * dy + 1e-4;
                let dist = dist_sq.sqrt();
                
                if dist < 5.0 {
                    let force = k_repulsion / dist_sq;
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;
                    
                    forces[i][0] += fx;
                    forces[i][1] += fy;
                    forces[j][0] -= fx;
                    forces[j][1] -= fy;
                }
            }
        }

        // Spring forces (bonds)
        for bond in &mol.inner.bonds {
            let u = bond.source_idx;
            let v = bond.target_idx;
            if u >= num_atoms || v >= num_atoms {
                continue;
            }
            let dx = coords[u][0] - coords[v][0];
            let dy = coords[u][1] - coords[v][1];
            let dist = (dx * dx + dy * dy + 1e-4).sqrt();
            
            let force = k_spring * (dist - d_zero);
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;

            forces[u][0] -= fx;
            forces[u][1] -= fy;
            forces[v][0] += fx;
            forces[v][1] += fy;
        }

        // Update positions
        for i in 0..num_atoms {
            coords[i][0] += forces[i][0] * dt;
            coords[i][1] += forces[i][1] * dt;
        }
    }

    // Save 2D coords
    let mut data = (*mol.inner).clone();
    data.coords_2d = Some(coords);
    mol.inner = Arc::new(data);
}

pub fn generate_3d_coords(mol: &mut RustMolecule) {
    let num_atoms = mol.num_atoms();
    if num_atoms == 0 {
        return;
    }

    // Distance Geometry (ETKDG-like inflation)
    // 1. Generate Distance Bounds Matrix
    let mut d_matrix = vec![vec![0.0; num_atoms]; num_atoms];
    
    // Build shortest path distances to approximate bounds
    let mut adj = vec![vec![1e9; num_atoms]; num_atoms];
    for i in 0..num_atoms {
        adj[i][i] = 0.0;
    }
    for bond in &mol.inner.bonds {
        let u = bond.source_idx;
        let v = bond.target_idx;
        if u < num_atoms && v < num_atoms {
            let len = match bond.bond_type {
                BondType::Single => 1.54,
                BondType::Double => 1.34,
                BondType::Triple => 1.20,
                BondType::Aromatic => 1.40,
            };
            adj[u][v] = len;
            adj[v][u] = len;
        }
    }

    // Floyd-Warshall to get approximate 3D distances
    for k in 0..num_atoms {
        for i in 0..num_atoms {
            for j in 0..num_atoms {
                if adj[i][k] + adj[k][j] < adj[i][j] {
                    adj[i][j] = adj[i][k] + adj[k][j];
                }
            }
        }
    }

    for i in 0..num_atoms {
        for j in 0..num_atoms {
            d_matrix[i][j] = if adj[i][j] > 1e6 { (i as f64 - j as f64).abs() * 1.54 } else { adj[i][j] };
        }
    }

    // 2. Metric Matrix (MDS Embedding)
    // For classical metric scaling, build Gram matrix B from squared distances
    let mut b = vec![vec![0.0; num_atoms]; num_atoms];
    for i in 0..num_atoms {
        for j in 0..num_atoms {
            let mut sum_ik = 0.0;
            let mut sum_jk = 0.0;
            let mut sum_all = 0.0;
            for k in 0..num_atoms {
                sum_ik += d_matrix[i][k] * d_matrix[i][k];
                sum_jk += d_matrix[j][k] * d_matrix[j][k];
                for l in 0..num_atoms {
                    sum_all += d_matrix[k][l] * d_matrix[k][l];
                }
            }
            let n = num_atoms as f64;
            b[i][j] = -0.5 * (d_matrix[i][j] * d_matrix[i][j] - sum_ik / n - sum_jk / n + sum_all / (n * n));
        }
    }

    // Embed into 3D using MDS approximation
    // Let's generate a simplified coordinates matrix using initial principal axes
    let mut coords = vec![[0.0, 0.0, 0.0]; num_atoms];
    for i in 0..num_atoms {
        let angle = (i as f64) * 2.39996; // Golden angle for spiral distribution on sphere
        let z = 1.0 - (i as f64 / (num_atoms as f64 - 1.0 + 1e-6)) * 2.0;
        let radius = (1.0 - z * z).max(0.0).sqrt();
        let dist = d_matrix[0][i];
        coords[i] = [
            angle.cos() * radius * dist,
            angle.sin() * radius * dist,
            z * dist,
        ];
    }

    // 3. Force-field minimization to satisfy distance bounds
    let minimization_iters = 150;
    let dt = 0.05;
    let k_bond = 0.4;

    for _ in 0..minimization_iters {
        let mut forces = vec![[0.0, 0.0, 0.0]; num_atoms];

        for i in 0..num_atoms {
            for j in 0..num_atoms {
                if i == j {
                    continue;
                }
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let dist = (dx * dx + dy * dy + dz * dz + 1e-4).sqrt();
                
                let target_dist = d_matrix[i][j];
                let force = k_bond * (dist - target_dist);
                
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;
                let fz = (dz / dist) * force;

                forces[i][0] -= fx;
                forces[i][1] -= fy;
                forces[i][2] -= fz;
            }
        }

        // Apply updates
        for i in 0..num_atoms {
            coords[i][0] += forces[i][0] * dt;
            coords[i][1] += forces[i][1] * dt;
            coords[i][2] += forces[i][2] * dt;
        }
    }

    // Save 3D coords
    let mut data = (*mol.inner).clone();
    data.coords_3d = Some(coords);
    mol.inner = Arc::new(data);
}
