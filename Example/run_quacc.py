from ase import io
from nsc_dft import dhbeefvdw_flow

# Read structure
atoms = io.read(f"inputs/POSCAR")

# Run job
results = dhbeefvdw_flow(atoms,kpts=[12,12,1],calc_dir="./")
print(results)
