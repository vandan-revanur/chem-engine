class Atom:
    pass

class Bond:
    pass

class BondType:
    pass

class RustMolecule:
    pass

def parse_smiles(smiles: str) -> RustMolecule: ...
def canonicalize(smiles: str) -> str: ...
def generate_2d_coords(mol: RustMolecule): ...
def generate_3d_coords(mol: RustMolecule): ...
def batch_parse_smiles(smiles: list[str]): ...
