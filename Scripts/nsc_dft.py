"""QuAcc recipes for NSC-DFAs in VASP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from monty.os.path import zpath
from pymatgen.io.vasp import Vasprun
from quacc import change_settings

from quacc import flow, job
from quacc.recipes.vasp._base import (
    run_and_summarize,
)

import gzip
import shutil

from quacc.wflow_tools.job_argument import Copy
from analysis_scripts import get_energy, _open_file

LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quacc.types import (
        SourceDirectory,
        VaspSchema,
    )

@flow
def dhbeefvdw_flow(
    atoms: Atoms,
    preset: str | None = "DefaultSetGGA",
    calc_dir: str | Path = "calc_dir",
    job1_kwargs: dict[str, Any] | None = None,
    job2_kwargs: dict[str, Any] | None = None,
    job3_kwargs: dict[str, Any] | None = None,
    job4_kwargs: dict[str, Any] | None = None,
    job5_kwargs: dict[str, Any] | None = None,
    job6_kwargs: dict[str, Any] | None = None,
    job7_kwargs: dict[str, Any] | None = None,
    job8_kwargs: dict[str, Any] | None = None,
    **calc_kwargs
) -> dict[str, VaspSchema]:
    """
    Perform hBEEF-vdW@BEEF-vdW calculation as follows:
        1. BEEF-vdW single-point calculation with WAVECAR output
        2. BEEF exchange-correlation only NSCF calculation with WAVECAR from step 1
        3. BEEF exchange-only NSCF calculation with WAVECAR from step 1
        4. EXX NSCF calculation with WAVECAR from step 1

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    job1_kwargs
        Custom kwargs for the first VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported.
    job2_kwargs
        Custom kwargs for the second VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported
    job3_kwargs
        Custom kwargs for the third VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported
    job4_kwargs
        Custom kwargs for the fourth VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported

    Returns
    -------
    dict[str, VaspSchema]
        Dictionary of results from each step.
    """
    
    Path(calc_dir).mkdir(parents=True, exist_ok=True)

    # Create RPA preset yaml file in calc_dir
    create_rpa_yaml(Path(calc_dir))
    rpa_preset_path = f"{calc_dir}/RPASet.yaml"
    calc_kwargs["gamma"] = True

    job1_kwargs = {**(calc_kwargs or {}), **(job1_kwargs or {})}
    job2_kwargs = {**(calc_kwargs or {}), **(job2_kwargs or {})}
    job3_kwargs = {**(calc_kwargs or {}), **(job3_kwargs or {})}
    job4_kwargs = {**(calc_kwargs or {}), **(job4_kwargs or {})}
    job5_kwargs = {**(calc_kwargs or {}), **(job5_kwargs or {})}
    job6_kwargs = {**(calc_kwargs or {}), **(job6_kwargs or {})}
    job7_kwargs = {**(calc_kwargs or {}), **(job7_kwargs or {})}
    job8_kwargs = {**(calc_kwargs or {}), **(job8_kwargs or {})}

    # Run first job
    if (Path(calc_dir, "01_beefxc_vdw", "OUTCAR.gz").exists() or Path(calc_dir, "01_beefxc_vdw", "OUTCAR").exists()) and (Path(calc_dir, "01_beefxc_vdw", "WAVECAR.gz").exists() or Path(calc_dir, "01_beefxc_vdw", "WAVECAR").exists()):
        LOGGER.info(f"OUTCAR and WAVECAR found in {calc_dir}/01_beefxc_vdw. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR and WAVECAR not found in {calc_dir}/01_beefxc_vdw. Running job.")

        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "01_beefxc_vdw",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
        }
        ):
            summary1 = job_1_beefxc_vdw(
                atoms,
                preset=preset,
                **job1_kwargs,
                use_custodian=False
            )

    # Run second job
    if (Path(calc_dir, "02_beefxc", "OUTCAR.gz").exists() or Path(calc_dir, "02_beefxc", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/02_beefxc. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/02_beefxc. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "02_beefxc",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):
            try:
                summary2 = job_2_beefxc(
                    atoms,
                    prev_dir=Path(calc_dir, "01_beefxc_vdw"),
                    preset=preset,
                    **job2_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "02_beefxc").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "02_beefxc")
                    if check_vasp_finish(Path(calc_dir, "02_beefxc")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/02_beefxc.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/02_beefxc. Please check for errors in the calculations.")

    # Run third job
    if (Path(calc_dir, "03_beefx", "OUTCAR.gz").exists() or Path(calc_dir, "03_beefx", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/03_beefx. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/03_beefx. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "03_beefx",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
            
        }
        ):
            try:
                summary3 = job_3_beefx(
                    atoms,
                    prev_dir=Path(calc_dir, "01_beefxc_vdw"),
                    preset=preset,
                    **job3_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "03_beefx").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "03_beefx")
                    if check_vasp_finish(Path(calc_dir, "03_beefx")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/03_beefx.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/03_beefx. Please check for errors in the calculations.")

    # Run fourth job
    if (Path(calc_dir, "04_exx", "OUTCAR.gz").exists() or Path(calc_dir, "04_exx", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/04_exx. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/04_exx. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "04_exx",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):
            try:  
                summary4 = job_4_exx(
                    atoms,
                    prev_dir=Path(calc_dir, "01_beefxc_vdw"),
                    preset=preset,
                    **job4_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "04_exx").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "04_exx")
                    if check_vasp_finish(Path(calc_dir, "04_exx")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/04_exx.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/04_exx. Please check for errors in the calculations.")

    # Run fifth job
    if (Path(calc_dir, "05_pbe_rpa", "OUTCAR.gz").exists() or Path(calc_dir, "05_pbe_rpa", "OUTCAR").exists()) and (Path(calc_dir, "05_pbe_rpa", "WAVECAR.gz").exists() or Path(calc_dir, "05_pbe_rpa", "WAVECAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/05_pbe_rpa. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/05_pbe_rpa. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "05_pbe_rpa",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):  
            summary5 = job_5_pbe_rpa(
                atoms,
                preset=rpa_preset_path,
                **job5_kwargs,
                use_custodian=False
            )

    # Run sixth job
    if (Path(calc_dir, "06_pbe_exx", "OUTCAR.gz").exists() or Path(calc_dir, "06_pbe_exx", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/06_pbe_exx. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/06_pbe_exx. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "06_pbe_exx",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):
            try:
                summary6 = job_6_exx(
                    atoms,
                    prev_dir=Path(calc_dir, "05_pbe_rpa"),
                    preset=rpa_preset_path,
                    **job6_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "06_pbe_exx").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "06_pbe_exx")
                    if check_vasp_finish(Path(calc_dir, "06_pbe_exx")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/06_pbe_exx.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/06_pbe_exx. Please check for errors in the calculations.")

    # Run seventh job
    if (Path(calc_dir, "07_bands", "OUTCAR.gz").exists() or Path(calc_dir, "07_bands", "OUTCAR").exists()) and (Path(calc_dir, "07_bands", "WAVECAR.gz").exists() or Path(calc_dir, "07_bands", "WAVECAR").exists()):
        LOGGER.info(f"OUTCAR and WAVECAR found in {calc_dir}/07_bands. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR and WAVECAR not found in {calc_dir}/07_bands. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "07_bands",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False,
        }
        ):
            try:
                summary7 = job_7_bands(
                    atoms,
                    prev_dir=Path(calc_dir, "05_pbe_rpa"),
                    preset=rpa_preset_path,
                    **job7_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "07_bands").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "07_bands")
                    if check_vasp_finish(Path(calc_dir, "07_bands")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/07_bands.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/07_bands. Please check for errors in the calculations.")

    # Run eighth job
    if (Path(calc_dir, "08_rpa", "OUTCAR.gz").exists() or Path(calc_dir, "08_rpa", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/08_rpa. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/08_rpa. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "08_rpa",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):
            try:
                summary8 = job_8_rpac(
                    atoms,
                    prev_dir=Path(calc_dir, "07_bands"),
                    preset=rpa_preset_path,
                    **job8_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "08_rpa").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "08_rpa")
                    if check_vasp_finish(Path(calc_dir, "08_rpa")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/08_rpa.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/08_rpa. Please check for errors in the calculations.")


    return analyze_dhbeefvdw_flow(calc_dir)

@flow
def hbeefvdw_flow(
    atoms: Atoms,
    preset: str | None = "DefaultSetGGA",
    calc_dir: str | Path = "calc_dir",
    job1_kwargs: dict[str, Any] | None = None,
    job2_kwargs: dict[str, Any] | None = None,
    job3_kwargs: dict[str, Any] | None = None,
    job4_kwargs: dict[str, Any] | None = None,
    **calc_kwargs
) -> dict[str, VaspSchema]:
    """
    Perform hBEEF-vdW@BEEF-vdW calculation as follows:
        1. BEEF-vdW single-point calculation with WAVECAR output
        2. BEEF exchange-correlation only NSCF calculation with WAVECAR from step 1
        3. BEEF exchange-only NSCF calculation with WAVECAR from step 1
        4. EXX NSCF calculation with WAVECAR from step 1

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    job1_kwargs
        Custom kwargs for the first VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported.
    job2_kwargs
        Custom kwargs for the second VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported
    job3_kwargs
        Custom kwargs for the third VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported
    job4_kwargs
        Custom kwargs for the fourth VASP calculation. Set a value to `None` to remove a pre-existing key entirely. For a list of available keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE Vasp calculator keyword arguments are supported

    Returns
    -------
    dict[str, VaspSchema]
        Dictionary of results from each step.
    """

    Path(calc_dir).mkdir(parents=True, exist_ok=True)

    job1_kwargs = {**(calc_kwargs or {}), **(job1_kwargs or {})}
    job2_kwargs = {**(calc_kwargs or {}), **(job2_kwargs or {})}
    job3_kwargs = {**(calc_kwargs or {}), **(job3_kwargs or {})}
    job4_kwargs = {**(calc_kwargs or {}), **(job4_kwargs or {})}

    # Run first job
    if (Path(calc_dir, "01_beefxc_vdw", "OUTCAR.gz").exists() or Path(calc_dir, "01_beefxc_vdw", "OUTCAR").exists()) and (Path(calc_dir, "01_beefxc_vdw", "WAVECAR.gz").exists() or Path(calc_dir, "01_beefxc_vdw", "WAVECAR").exists()):
        LOGGER.info(f"OUTCAR and WAVECAR found in {calc_dir}/01_beefxc_vdw. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR and WAVECAR not found in {calc_dir}/01_beefxc_vdw. Running job.")

        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "01_beefxc_vdw",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
        }
        ):
            summary1 = job_1_beefxc_vdw(
                atoms,
                preset=preset,
                **job1_kwargs,
                use_custodian=False
            )

    # Run second job
    if (Path(calc_dir, "02_beefxc", "OUTCAR.gz").exists() or Path(calc_dir, "02_beefxc", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/02_beefxc. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/02_beefxc. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "02_beefxc",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):
            try:
                summary2 = job_2_beefxc(
                    atoms,
                    prev_dir=Path(calc_dir, "01_beefxc_vdw"),
                    preset=preset,
                    **job2_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "02_beefxc").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "02_beefxc")
                    if check_vasp_finish(Path(calc_dir, "02_beefxc")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/02_beefxc.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/02_beefxc. Please check for errors in the calculations.")

    # Run third job
    if (Path(calc_dir, "03_beefx", "OUTCAR.gz").exists() or Path(calc_dir, "03_beefx", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/03_beefx. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/03_beefx. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "03_beefx",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
            
        }
        ):
            try: 
                summary3 = job_3_beefx(
                    atoms,
                    prev_dir=Path(calc_dir, "01_beefxc_vdw"),
                    preset=preset,
                    **job3_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "03_beefx").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "03_beefx")
                    if check_vasp_finish(Path(calc_dir, "03_beefx")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/03_beefx.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/03_beefx. Please check for errors in the calculations.")

    # Run fourth job
    if (Path(calc_dir, "04_exx", "OUTCAR.gz").exists() or Path(calc_dir, "04_exx", "OUTCAR").exists()):
        LOGGER.info(f"OUTCAR found in {calc_dir}/04_exx. Skipping job.")
    else:
        LOGGER.info(f"OUTCAR not found in {calc_dir}/04_exx. Running job.")
        with change_settings(
        {
            "RESULTS_DIR": Path(calc_dir) / "04_exx",
            "CREATE_UNIQUE_DIR": False,
            "GZIP_FILES": True,
            "CHECK_CONVERGENCE": False, 
        }
        ):
            try:  
                summary4 = job_4_exx(
                    atoms,
                    prev_dir=Path(calc_dir, "01_beefxc_vdw"),
                    preset=preset,
                    **job4_kwargs,
                    use_custodian=False
                )
            except:
                failed_dirs = sorted(Path(calc_dir, "04_exx").glob("failed-quacc*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if failed_dirs:
                    most_recent_failed_dir = failed_dirs[0]
                    move_and_gzip_all(most_recent_failed_dir, Path(calc_dir) / "04_exx")
                    if check_vasp_finish(Path(calc_dir, "04_exx")):
                        LOGGER.info(f"Calculation in {most_recent_failed_dir} has finished. Moved and gzipped files in {calc_dir}/04_exx.")
                    else:
                        raise RuntimeError(f"Calculation in {most_recent_failed_dir} has not finished. Please check for errors in the calculations.")
                else:
                    raise RuntimeError(f"No failed-quacc* directories found in {calc_dir}/04_exx. Please check for errors in the calculations.")

    return analyze_hbeefvdw_flow(calc_dir)

@job
def job_1_beefxc_vdw(
    atoms: Atoms,
    preset: str | None = "DefaultSetGGA",
    copy_files: SourceDirectory | Copy | None = None,
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a single-point calculation with BEEF-vdW and summarize the results.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "encut": 550,
        "lasph": True,
        "ismear": -1,
        "sigma": 0.10,
        "gga": "LIBXC",
        "libxc1": "GGA_XC_BEEFVDW",
        "luse_vdw": True,
        "zab_vdw": -1.8867,
        "prec": "Accurate",
        "algo": "All",
        "lreal": "Auto",
        "ediff": 1e-8,
        "isym": 0,
        "nsw": 0,
        "lcharg": False,
        "lwave": True,
        "ivdw": 0
    }
    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP BEEF-vdW Static"} | (additional_fields or {}),
        copy_files=copy_files,
    )

@job
def job_2_beefxc(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    preset: str | None = "DefaultSetGGA",
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a non-self-consistent field (NSCF) calculation for the BEEF XC
    functional on top of wave-functions from a previous directory.

    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a WAVECAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "encut": 550,
        "lasph": True,
        "ismear": -1,
        "sigma": 0.10,
        "gga": "LIBXC",
        "libxc1": "GGA_XC_BEEFVDW",
        "luse_vdw": False,
        "prec": "Accurate",
        "algo": "Eigenval",
        "lreal": "Auto",
        "istart": 1,
        "nelm": 1,
        "ediff": 1e-8,
        "isym": 0,
        "nsw": 0,
        "lcharg": False,
        "lwave": False,
        "ivdw": 0
    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP BEEF (no vdW) Non-SCF"} | (additional_fields or {}),
        copy_files={prev_dir: ["WAVECAR*"]} if prev_dir else None,
    )

@job
def job_3_beefx(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    preset: str | None = "DefaultSetGGA",
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a non-self-consistent field (NSCF) calculation for the BEEF
    exchange-only functional on top of wave-functions from a previous directory.
    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a CHGCAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "encut": 550,
        "lasph": True,
        "ismear": -1,
        "sigma": 0.10,
        "gga": "LIBXC",
        "libxc1": "GGA_X_BEEFVDW",
        "luse_vdw": False,
        "prec": "Accurate",
        "algo": "Eigenval",
        "lreal": "Auto",
        "istart": 1,
        "nelm": 1,
        "ediff": 1e-8,
        "isym": 0,
        "nsw": 0,
        "lcharg": False,
        "lwave": False,
        "ivdw": 0
    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP BEEF exchange Non-SCF"} | (additional_fields or {}),
        copy_files={prev_dir: ["WAVECAR*"]} if prev_dir else None,
    )

@job
def job_4_exx(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    preset: str | None = "DefaultSetGGA",
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a non-self-consistent field (NSCF) calculation.

    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a CHGCAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "encut": 550,
        "lasph": True,
        "ispin": 1,
        "ismear": -1,
        "sigma": 0.10,
        "gga": "PE",
        "lhfcalc": True,
        "aexx": 1.0,
        "hfscreen": 0.3,
        "prec": "Accurate",
        "algo": "Eigenval",
        "lreal": "Auto",
        "istart": 1,
        "nelm": 1,
        "ediff": 1e-8,
        "isym": 0,
        "nsw": 0,
        "lcharg": False,
        "lwave": False,
        "ivdw": 0

    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP EXX Non-SCF"} | (additional_fields or {}),
        copy_files={prev_dir: ["WAVECAR*"]} if prev_dir else None,
    )

@job
def job_5_pbe_rpa(
    atoms: Atoms,
    preset: str | None = "DefaultSetGGA",
    copy_files: SourceDirectory | Copy | None = None,
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a single-point calculation with PBE and summarize the results.

    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a CHGCAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "ismear": -1,
        "sigma": 0.10,
        "ediff": 1e-8,
        "encut": 310,
        "prec": "Normal",
        "istart": 0,
        "lmaxfockae": 4,
        "loptics": False,
        "algo": "All",
        "isym": 0,
        "lcharg": False,
        "lwave": True,
        "ivdw": 0

    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP PBE for RPA"} | (additional_fields or {}),
        copy_files=copy_files,
    )

@job
def job_6_exx(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    preset: str | None = "DefaultSetGGA",
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a non-self-consistent field (NSCF) calculation for exact exchange (EXX)

    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a CHGCAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "ismear": -1,
        "sigma": 0.10,
        "algo": "EIGENVAL",
        "nelm": 1,
        "ediff": 1e-8,
        "encut": 310,
        "prec": "Normal",
        "precfock": "Normal",
        "lmaxfockae": 4,
        "lwave": False,
        "lhfcalc": True,
        "aexx": 1.0,
        "isym": 0,
        "ivdw": 0

    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP EXX Non-SCF"} | (additional_fields or {}),
        copy_files={prev_dir: ["WAVECAR*"]} if prev_dir else None,
    )

@job
def job_7_bands(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    preset: str | None = "DefaultSetGGA",
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out an exact diagonalization to get all the bands for RPA correlation.
    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a CHGCAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    # Get the maximum number of plane-waves from the OUTCAR
    if Path(prev_dir, "OUTCAR.gz").exists() and Path(prev_dir, "OUTCAR").exists():
        outcar_path = Path(prev_dir, "OUTCAR.gz")
    elif Path(prev_dir, "OUTCAR.gz").exists():
        outcar_path = Path(prev_dir, "OUTCAR.gz")
    elif Path(prev_dir, "OUTCAR").exists():
        outcar_path = Path(prev_dir, "OUTCAR")
    else:
        raise FileNotFoundError(f"Neither OUTCAR nor OUTCAR.gz found in {prev_dir}")

    with _open_file(outcar_path, mode="rt", encoding="ISO-8859-1") as f:
        for line in f:
            if "maximum number of plane-waves:" in line:
                max_planewaves = int(line.split()[-1])
                break

    calc_defaults = {
        "ismear": -1,
        "sigma": 0.10,
        "algo": "Exact",
        "loptics": False,
        "ediff": 1e-8,
        "encut": 310,
        "prec": "Normal",
        "lmaxfockae": 4,
        "nelm": 1,
        "nbands": max_planewaves,
        "isym": 0,
        "ivdw": 0

    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP Exact Diagonalization"} | (additional_fields or {}),
        copy_files={prev_dir: ["WAVECAR*"]} if prev_dir else None,
    )

@job
def job_8_rpac(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    preset: str | None = "DefaultSetGGA",
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out RPA correlation calculation.

    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir
        Directory of the prior job. Must contain a CHGCAR and vasprun.xml file.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][]. All of the ASE
        Vasp calculator keyword arguments are supported.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """

    # Get the maximum number of plane-waves from the OUTCAR
    if Path(prev_dir, "OUTCAR.gz").exists() and Path(prev_dir, "OUTCAR").exists():
        outcar_path = Path(prev_dir, "OUTCAR.gz")
    elif Path(prev_dir, "OUTCAR.gz").exists():
        outcar_path = Path(prev_dir, "OUTCAR.gz")
    elif Path(prev_dir, "OUTCAR").exists():
        outcar_path = Path(prev_dir, "OUTCAR")
    else:
        raise FileNotFoundError(f"Neither OUTCAR nor OUTCAR.gz found in {prev_dir}")

    with _open_file(outcar_path, mode="rt", encoding="ISO-8859-1") as f:
        for line in f:
            if "maximum number of plane-waves:" in line:
                max_planewaves = int(line.split()[-1])
                break

    calc_kwargs["ncore"] = 1
    calc_kwargs["kpar"] = 1

    calc_defaults = {
        "ismear": -1,
        "sigma": 0.10,
        "algo": "ACFDTR",
        "loptics": False,
        "ediff": 1e-8,
        "encut": 310,
        "prec": "Normal",
        "precfock": "Fast",
        "lmaxfockae": 4,
        "nbands": max_planewaves,
        "lfinite_temperature": True,
        "isym": 0,
        "ivdw": 0

    }

    return run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP RPA correlation"} | (additional_fields or {}),
        copy_files={prev_dir: ["WAVECAR*"]} if prev_dir else None,
    )

def analyze_dhbeefvdw_flow(calc_dir: str | Path) -> dict[str, float]:
    """
    Analyze the results from the dhBEEF-vdW flow to extract the energies.

    Parameters
    ----------
    calc_dir
    Directory where the calculations were run.

    Returns
    -------
    dict[str, float]
        Dictionary containing the energies from each step.
    """

    calc_energies = { method: 0 for method in ["beef_xc_vdw", "beef_xc", "beef_x", "exx", "hbeef_vdw","pbe","pbe_exx","rpac","rpa","dhbeef_vdw"] }

    # Step 1: BEEF-vdW
    calc_energies["beef_xc_vdw"] = get_energy(Path(calc_dir, "01_beefxc_vdw",'OUTCAR.gz'), code="vasp")

    # Step 2: BEEF XC only
    calc_energies["beef_xc"] = get_energy(Path(calc_dir, "02_beefxc",'OUTCAR.gz'), code="vasp")

    # Step 3: BEEF exchange only
    calc_energies["beef_x"] = get_energy(Path(calc_dir, "03_beefx",'OUTCAR.gz'), code="vasp")

    # Step 4: EXX
    calc_energies["exx"] = get_energy(Path(calc_dir, "04_exx",'OUTCAR.gz'), code="vasp")

    # Calculate hBEEF-vdW energy
    nlc_energy = calc_energies["beef_xc_vdw"] - calc_energies["beef_xc"]
    beefc_energy = calc_energies["beef_xc"] - calc_energies["beef_x"]
    beefx_energy = calc_energies["beef_x"]
    exx_energy = calc_energies["exx"]
    h_x_frac = 0.175
    hbeefvdw_energy = h_x_frac * exx_energy + (1-h_x_frac)*beefx_energy + beefc_energy + nlc_energy
    calc_energies["hbeef_vdw"] = hbeefvdw_energy

    # Step 5: PBE for RPA
    calc_energies["pbe"] = get_energy(Path(calc_dir, "05_pbe_rpa",'OUTCAR.gz'), code="vasp")

    # Step 6: EXX for RPA
    calc_energies["pbe_exx"] = get_energy(Path(calc_dir, "06_pbe_exx",'OUTCAR.gz'), code="vasp")

    # Step 8: RPA correlation energy
    calc_energies["rpac"] = get_energy(Path(calc_dir, "08_rpa",'OUTCAR.gz'), code="vasp_rpa")

    # Calculate RPA energy
    calc_energies["rpa"] = calc_energies["pbe_exx"] + calc_energies["rpac"]

    # Calculate dhBEEF-vdW energy
    rpac_energy = calc_energies["rpac"]
    dh_x_frac = 0.25
    dh_c_frac = 0.15
    calc_energies["dhbeef_vdw"] = dh_x_frac * exx_energy + (1-dh_x_frac)*beefx_energy + dh_c_frac*rpac_energy + (1-dh_c_frac)*beefc_energy + nlc_energy

    return calc_energies


def analyze_hbeefvdw_flow(calc_dir: str | Path) -> dict[str, float]:
    """
    Analyze the results from the hBEEF-vdW flow to extract the energies.

    Parameters
    ----------
    calc_dir
    Directory where the calculations were run.

    Returns
    -------
    dict[str, float]
        Dictionary containing the energies from each step.
    """
    calc_energies = { method: 0 for method in ["beef_xc_vdw", "beef_xc", "beef_x", "exx", "hbeef_vdw"] }

    # Step 1: BEEF-vdW
    calc_energies["beef_xc_vdw"] = get_energy(Path(calc_dir, "01_beefxc_vdw",'OUTCAR.gz'), code="vasp")

    # Step 2: BEEF XC only
    calc_energies["beef_xc"] = get_energy(Path(calc_dir, "02_beefxc",'OUTCAR.gz'), code="vasp")

    # Step 3: BEEF exchange only
    calc_energies["beef_x"] = get_energy(Path(calc_dir, "03_beefx",'OUTCAR.gz'), code="vasp")

    # Step 4: EXX
    calc_energies["exx"] = get_energy(Path(calc_dir, "04_exx",'OUTCAR.gz'), code="vasp")

    # Calculate hBEEF-vdW energy
    nlc_energy = calc_energies["beef_xc_vdw"] - calc_energies["beef_xc"]
    beefc_energy = calc_energies["beef_xc"] - calc_energies["beef_x"]
    beefx_energy = calc_energies["beef_x"]
    exx_energy = calc_energies["exx"]
    h_x_frac = 0.175
    hbeefvdw_energy = h_x_frac * exx_energy + (1-h_x_frac)*beefx_energy + beefc_energy + nlc_energy
    calc_energies["hbeef_vdw"] = hbeefvdw_energy

    return calc_energies

def move_and_gzip_all(src_dir, dst_dir, *, keep_name=True):
    """
    Move all files from src_dir to dst_dir, gzip-compressing them in the process. Original files are removed after successful compression.

    Parameters
    ----------
    src_dir
        Source directory containing files to move and gzip.
    dst_dir        
        Destination directory to move gzipped files to.
    keep_name
        If True, gzipped files will have the same name as the original with a .gz extension. If False, gzipped files will have the same stem as the original but with a .gz extension (i.e. original "WAVECAR" becomes "WAVECAR.gz" if keep_name is True, but becomes "WAVECAR.gz" if keep_name is False).
        
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.iterdir():
        if not p.is_file():
            continue  # skip subdirs, symlinks, etc.

        gz_name = p.name + ".gz" if keep_name else p.stem + ".gz"
        out_path = dst_dir / gz_name

        # stream-compress to destination
        with p.open("rb") as f_in, gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # remove original after successful write
        p.unlink()

def check_vasp_finish(calc_dir: str | Path) -> bool:
    """
    Check if a VASP calculation has finished by looking for the presence of an OUTCAR or OUTCAR.gz file.

    Parameters
    ----------
    calc_dir
        Directory where the VASP calculation was run.

    Returns
    -------
    bool
        True if OUTCAR or OUTCAR.gz is found, False otherwise.
    """
    calc_dir = Path(calc_dir)

    # Check for OUTCAR.gz first, then OUTCAR, and if exist, confirm that "Total CPU time used" is present in the OUTCAR to ensure the calculation finished properly

    if (calc_dir / "OUTCAR.gz").exists():
        outcar_path = calc_dir / "OUTCAR.gz"
    elif (calc_dir / "OUTCAR").exists():
        outcar_path = calc_dir / "OUTCAR"
    else:
        return False
    
    with _open_file(outcar_path, mode="rt", encoding="ISO-8859-1") as f:
        lines = f.readlines()[-100:]
        for line in lines:
            if "Total CPU time used" in line:
                return True
            
def create_rpa_yaml(filedir: str | Path):
    """
    Create a YAML file containing the POTCAR files for the RPA calculation.

    Parameters
    ----------
    filedir
        Directory to save the YAML file in.
    energies
        Dictionary of energies to save in the YAML file.
    """

    gw_potcar_dict = {
    'Ac': 'Ac', 'Ag': 'Ag_GW', 'Al': 'Al_GW', 'Am': 'Am',
    'Ar': 'Ar_GW', 'As': 'As_GW', 'At': 'At_d_GW', 'Au': 'Au_GW',
    'B': 'B_GW', 'Ba': 'Ba_sv_GW', 'Be': 'Be_GW', 'Bi': 'Bi_GW',
    'Br': 'Br_GW', 'C': 'C_GW', 'Ca': 'Ca_sv_GW', 'Cd': 'Cd_GW',
    'Ce': 'Ce_GW', 'Cf': 'Cf', 'Cl': 'Cl_GW', 'Cm': 'Cm',
    'Co': 'Co_GW', 'Cr': 'Cr_sv_GW', 'Cs': 'Cs_sv_GW', 'Cu': 'Cu_GW',
    'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu_3', 'F': 'F_GW',
    'Fe': 'Fe_GW', 'Fr': 'Fr_sv', 'Ga': 'Ga_GW', 'Gd': 'Gd_3',
    'Ge': 'Ge_GW', 'H': 'H_GW', 'He': 'He_GW', 'Hf': 'Hf_sv_GW',
    'Hg': 'Hg_sv_GW', 'Ho': 'Ho_3', 'I': 'I_GW', 'In': 'In_d_GW',
    'Ir': 'Ir_sv_GW', 'K': 'K_sv_GW', 'Kr': 'Kr_GW', 'La': 'La_GW',
    'Li': 'Li_GW', 'Lu': 'Lu_3', 'Mg': 'Mg_GW', 'Mn': 'Mn_GW',
    'Mo': 'Mo_sv_GW', 'N': 'N_GW', 'Na': 'Na_sv_GW', 'Nb': 'Nb_sv_GW',
    'Nd': 'Nd_3', 'Ne': 'Ne_GW', 'Ni': 'Ni_GW', 'Np': 'Np',
    'O': 'O_GW', 'Os': 'Os_sv_GW', 'P': 'P_GW', 'Pa': 'Pa',
    'Pb': 'Pb_d_GW', 'Pd': 'Pd_GW', 'Pm': 'Pm_3', 'Po': 'Po_d_GW',
    'Pr': 'Pr_3', 'Pt': 'Pt_GW', 'Pu': 'Pu', 'Ra': 'Ra_sv',
    'Rb': 'Rb_sv_GW', 'Re': 'Re_sv_GW', 'Rh': 'Rh_GW', 'Rn': 'Rn_d_GW',
    'Ru': 'Ru_sv_GW', 'S': 'S_GW', 'Sb': 'Sb_GW', 'Sc': 'Sc_sv_GW',
    'Se': 'Se_GW', 'Si': 'Si_GW', 'Sm': 'Sm_3', 'Sn': 'Sn_d_GW',
    'Sr': 'Sr_sv_GW', 'Ta': 'Ta_sv_GW', 'Tb': 'Tb_3', 'Tc': 'Tc_sv_GW',
    'Te': 'Te_GW', 'Th': 'Th', 'Ti': 'Ti_sv_GW', 'Tl': 'Tl_sv_GW',
    'Tm': 'Tm_3', 'U': 'U', 'V': 'V_sv_GW', 'W': 'W_sv_GW',
    'Xe': 'Xe_GW', 'Y': 'Y_sv_GW', 'Yb': 'Yb_3', 'Zn': 'Zn_GW',
    'Zr': 'Zr_sv_GW',
    }

    with open(Path(filedir, "RPASet.yaml"), 'w') as f:
        f.write('inputs:\n')
        f.write('  pp_version: "64"\n')
        f.write('  pp: "PBE"\n')
        f.write('  setups:\n')
        for element, potential in gw_potcar_dict.items():
            f.write(f'    {element}: {potential}\n')
    
