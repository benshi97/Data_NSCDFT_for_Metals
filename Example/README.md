# Example: CO molecule

This folder demonstrates how to run the **hBEEF-vdW / dhBEEF-vdW workflows** for a CO molecule using either the Python (`quacc`) workflow or the standalone Bash workflow.

---

## Contents

```
01_beefxc_vdw  02_beefxc  03_beefx  04_exx
05_pbe_rpa     06_pbe_exx 07_bands  08_rpa
inputs         run_quacc.py
```

- `01_beefxc_vdw` … `08_rpa` — step directories used by the workflows  
- `inputs/` — required VASP input files for this example  
- `run_quacc.py` — Python script that runs the `dhbeefvdw_flow` and prints the analyzed energies  

---

## How to Run

### Option 1 — Python (`quacc` workflow)

```bash
python run_quacc.py
```

### Option 2 — Bash workflow

```bash
nsc_dft.sh "" dhbeefvdw
```

The first argument is the MPI launcher command. Using `""` runs serially (useful for testing), but you can also provide a launcher such as:

```bash
nsc_dft.sh "srun --distribution=block:block --hint=nomultithread" dhbeefvdw
```

Both approaches will run (or analyze) the calculations stored in the step directories.

---

## Example Output

After completion, the analyzed energies are printed:

```
----------------------------------------------------------------
beef_xc_vdw:   -12.0989351300
beef_xc:       -14.3761905700
beef_x:        -13.4335696200
exx:           -25.2017108900
hbeef_vdw:     -14.1583598522
----------------------------------------------------------------
pbe:           -13.8190145900
pbe_exx:       -30.6672269900
rpac:          -15.1756883678
rpa:           -45.8429153578
----------------------------------------------------------------
dhbeef_vdw:    -17.1759305602
----------------------------------------------------------------
```

All energies are reported in **electron volts (eV)** for the CO molecule.
