use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod atom;
pub mod bond;
pub mod molecule;
pub mod algorithms {
    pub mod smiles;
    pub mod layout;
}

use crate::atom::Atom;
use crate::bond::{Bond, BondType};
use crate::molecule::RustMolecule;

#[pyfunction]
pub fn parse_smiles(smiles: &str) -> PyResult<RustMolecule> {
    crate::algorithms::smiles::parse_smiles(smiles)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

#[pyfunction]
pub fn canonicalize(mol: &RustMolecule) -> PyResult<String> {
    Ok(crate::algorithms::smiles::canonicalize(mol))
}

#[pyfunction]
pub fn generate_2d_coords(mut mol: RustMolecule) -> PyResult<RustMolecule> {
    crate::algorithms::layout::generate_2d_coords(&mut mol);
    Ok(mol)
}

#[pyfunction]
pub fn generate_3d_coords(mut mol: RustMolecule) -> PyResult<RustMolecule> {
    crate::algorithms::layout::generate_3d_coords(&mut mol);
    Ok(mol)
}

// FR-4: Parallel Engine
#[pyfunction]
pub fn batch_parse_smiles(py: Python<'_>, smiles_list: Vec<String>) -> PyResult<Vec<RustMolecule>> {
    py.allow_threads(|| {
        use rayon::prelude::*;
        let results: Result<Vec<RustMolecule>, String> = smiles_list
            .into_par_iter()
            .map(|s| crate::algorithms::smiles::parse_smiles(&s))
            .collect();
        results.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    })
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<Bond>()?;
    m.add_class::<BondType>()?;
    m.add_class::<RustMolecule>()?;

    m.add_function(wrap_pyfunction!(parse_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize, m)?)?;
    m.add_function(wrap_pyfunction!(generate_2d_coords, m)?)?;
    m.add_function(wrap_pyfunction!(generate_3d_coords, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parse_smiles, m)?)?;

    Ok(())
}
