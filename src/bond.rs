use pyo3::prelude::*;
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondType {
    Single = 1,
    Double = 2,
    Triple = 3,
    Aromatic = 4,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Bond {
    #[pyo3(get, set)]
    pub source_idx: usize,
    #[pyo3(get, set)]
    pub target_idx: usize,
    #[pyo3(get, set)]
    pub bond_type: BondType,
}

#[pymethods]
impl Bond {
    #[new]
    pub fn new(source_idx: usize, target_idx: usize, bond_type: BondType) -> Self {
        Self {
            source_idx,
            target_idx,
            bond_type,
        }
    }
}
