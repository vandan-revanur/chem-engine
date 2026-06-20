use pyo3::prelude::*;
#[pyclass]
#[derive(Debug, Clone)]
pub struct Atom {
    #[pyo3(get, set)]
    pub atomic_number: u16,
    #[pyo3(get, set)]
    pub formal_charge: i8,
    #[pyo3(get, set)]
    pub implicit_valence: u8,
    #[pyo3(get, set)]
    pub explicit_valence: u8,
    #[pyo3(get, set)]
    pub num_explicit_hs: u8,
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub is_aromatic: bool,
}

#[pymethods]
impl Atom {
    #[new]
    #[pyo3(signature = (atomic_number, formal_charge=0, num_explicit_hs=0, is_aromatic=false))]
    pub fn new(atomic_number: u16, formal_charge: i8, num_explicit_hs: u8, is_aromatic: bool) -> Self {
        let symbol = match atomic_number {
            1 => "H",
            5 => "B",
            6 => "C",
            7 => "N",
            8 => "O",
            9 => "F",
            15 => "P",
            16 => "S",
            17 => "Cl",
            35 => "Br",
            53 => "I",
            _ => "?",
        }.to_string();

        Self {
            atomic_number,
            formal_charge,
            implicit_valence: 0,
            explicit_valence: 0,
            num_explicit_hs,
            symbol,
            is_aromatic,
        }
    }

    #[getter]
    pub fn get_mass(&self) -> f64 {
        match self.atomic_number {
            1 => 1.008,
            5 => 10.811,
            6 => 12.011,
            7 => 14.007,
            8 => 15.999,
            9 => 18.998,
            15 => 30.974,
            16 => 32.06,
            17 => 35.45,
            35 => 79.904,
            53 => 126.90,
            _ => 0.0,
        }
    }
}
