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
            0  => "*",
            1  => "H",   2  => "He",  3  => "Li",  4  => "Be",  5  => "B",
            6  => "C",   7  => "N",   8  => "O",   9  => "F",  10  => "Ne",
            11 => "Na",  12 => "Mg",  13 => "Al",  14 => "Si",  15 => "P",
            16 => "S",   17 => "Cl",  18 => "Ar",  19 => "K",   20 => "Ca",
            21 => "Sc",  22 => "Ti",  23 => "V",   24 => "Cr",  25 => "Mn",
            26 => "Fe",  27 => "Co",  28 => "Ni",  29 => "Cu",  30 => "Zn",
            31 => "Ga",  32 => "Ge",  33 => "As",  34 => "Se",  35 => "Br",
            36 => "Kr",  37 => "Rb",  38 => "Sr",  39 => "Y",   40 => "Zr",
            41 => "Nb",  42 => "Mo",  43 => "Tc",  44 => "Ru",  45 => "Rh",
            46 => "Pd",  47 => "Ag",  48 => "Cd",  49 => "In",  50 => "Sn",
            51 => "Sb",  52 => "Te",  53 => "I",   54 => "Xe",  55 => "Cs",
            56 => "Ba",  57 => "La",  72 => "Hf",  73 => "Ta",  74 => "W",
            75 => "Re",  76 => "Os",  77 => "Ir",  78 => "Pt",  79 => "Au",
            80 => "Hg",  81 => "Tl",  82 => "Pb",  83 => "Bi",
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
            1  => 1.008,    2  => 4.003,    3  => 6.941,    4  => 9.012,
            5  => 10.811,   6  => 12.011,   7  => 14.007,   8  => 15.999,
            9  => 18.998,   10 => 20.180,   11 => 22.990,   12 => 24.305,
            13 => 26.982,   14 => 28.086,   15 => 30.974,   16 => 32.06,
            17 => 35.45,    18 => 39.948,   19 => 39.098,   20 => 40.078,
            21 => 44.956,   22 => 47.867,   23 => 50.942,   24 => 51.996,
            25 => 54.938,   26 => 55.845,   27 => 58.933,   28 => 58.693,
            29 => 63.546,   30 => 65.38,    31 => 69.723,   32 => 72.630,
            33 => 74.922,   34 => 78.971,   35 => 79.904,   36 => 83.798,
            37 => 85.468,   38 => 87.62,    39 => 88.906,   40 => 91.224,
            41 => 92.906,   42 => 95.95,    43 => 98.0,     44 => 101.07,
            45 => 102.906,  46 => 106.42,   47 => 107.868,  48 => 112.414,
            49 => 114.818,  50 => 118.710,  51 => 121.760,  52 => 127.60,
            53 => 126.904,  54 => 131.293,  55 => 132.905,  56 => 137.327,
            57 => 138.905,  72 => 178.49,   73 => 180.948,  74 => 183.84,
            75 => 186.207,  76 => 190.23,   77 => 192.217,  78 => 195.084,
            79 => 196.967,  80 => 200.592,  81 => 204.38,   82 => 207.2,
            83 => 208.980,
            _ => 0.0,
        }
    }
}
