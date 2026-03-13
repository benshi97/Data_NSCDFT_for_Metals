#!/usr/bin/env python
# coding: utf-8

# Description: Contains functions for reading and plotting data from MRCC, ORCA and VASP calculations.

import numpy as np
from datetime import datetime
import pandas as pd
import gzip
from scipy.interpolate import UnivariateSpline
from ase.io import read
from pathlib import Path

import re

# Define units
kB = 8.617330337217213e-05
mol = 6.022140857e+23
kcal = 2.611447418269555e+22
kJ = 6.241509125883258e+21
Hartree = 27.211386024367243
Bohr = 0.5291772105638411

# Some basic conversion factors
cm1_to_eV = 1 / 8065.54429
hundredcm1 = 100 * cm1_to_eV * 1000
kcalmol_to_meV = kcal / mol * 1000
kjmol_to_meV = kJ / mol * 1000
mha_to_meV = Hartree

def read_outcar_unit_cell(filename):
    """
    Read lattice parameters from a VASP OUTCAR file.
    
    Parameters
    ----------
    filename : str
        The name of the file to read lattice parameters from.
        
    Returns
    -------
    float
        3x3 numpy array of unit cell parameters.
    """

    # Read the OUTCAR file
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find the line with the lattice parameters
    for i, line in enumerate(lines):
        if "Lattice vectors" in line:
            unit_cell_parameters_str = lines[i + 2 : i + 5]
            break

    # Convert the lattice parameters to a 3x3 numpy array
    unit_cell_parameters = np.array(
        [
            [float(x[:-1]) for x in line.split()[3:]]
            for line in unit_cell_parameters_str
        ]
    )

    return unit_cell_parameters


# Getting the cost of the DFT calculations
def get_vasp_walltime(filename):
    """
    Reads the walltime from the OUTCAR file.
        
    Parameters
    ----------
    filename : str
        The location of the 'OUTCAR' file to read from.
        
    Returns
    -------
    float
        The walltime in seconds.
        
    Notes
    -----
    This function reads the 'OUTCAR' file and extracts the total walltime taken by the VASP calculation.
    """

    f = _open_file(filename)
    a = f.readlines()
    # Search for the line with "Elapsed time (sec):" string and get the last column of the line
    for line in a:
        if "Elapsed time (sec):" in line:
            total_time = float(line.split()[-1])
            break
    f.close()
    return total_time

def get_vasp_looptime(filename):
    """
    Reads the time for a single loop from the OUTCAR file.
        
    Parameters
    ----------
    filename : str
        The location of the 'OUTCAR' file to read from.
        
    Returns
    -------
    float
        The walltime in seconds.
        
    Notes
    -----
    This function reads the 'OUTCAR' file and extracts the time taken for a single SCF loop in a VASP calculation.
    """

    f = open(filename)
    a = f.readlines()
    loop_times = []
    # Search for the line with "Elapsed time (sec):" string and get the last column of the line
    for line in a:
        if "LOOP:" in line:
            loop_times += [float(line.split()[-1].replace("time", ""))]
    f.close()
    return np.mean(loop_times)


def _open_file(filename, mode="rt", encoding=None):
    """Open regular or gzipped file transparently."""
    filename = Path(filename)
    if filename.suffix == ".gz":
        return gzip.open(filename, mode=mode, encoding=encoding)
    return open(filename, mode=mode, encoding=encoding)


def get_energy(filename, method="ccsdt", code="vasp"):
    """
    Function to parse the energy from an output file.

    Parameters
    ----------
    filename : str
        The location of the output file to read from.
    method : str
        The type of method to read (used for MRCC).
    code : str
        The code format. Options: 'mrcc', 'vasp', 'vasp_rpa'

    Returns
    -------
    float
        The energy in the original units.
    """

    if code == "mrcc":
        if method == "rpa_corr":
            search_word = "DF-dRPA correlation energy [au]:"
        elif method == "rpa_exx":
            search_word = "Reference energy [au]:"
        else:
            return 0.0  # unsupported method

        with _open_file(filename, mode="rt") as fp:
            lines = [line for line in fp if search_word in line]

        return float(lines[-1].split()[-1]) if lines else 0.0

    elif code == "vasp":
        search_word = "energy  without entropy="
        with _open_file(filename, mode="rt", encoding="ISO-8859-1") as fp:
            lines = [line for line in fp if search_word in line]

        return float(lines[-1].split()[-1]) if lines else 0.0

    elif code == "vasp_rpa":
        search_word = "converged value"
        with _open_file(filename, mode="rt", encoding="ISO-8859-1") as fp:
            lines = [line for line in fp if search_word in line]

        return float(lines[-1].split()[-2]) if lines else 0.0

    return 0.0


def get_cbs(
    hf_X,
    corr_X,
    hf_Y,
    corr_Y,
    X=2,
    Y=3,
    family="cc",
    convert_Hartree=False,
    shift=0.0,
    output=True,
):
    """
    Function to perform basis set extrapolation of HF and correlation energies for both the cc-pVXZ and def2-XZVP basis sets
    
    Parameters
    ----------
    hf_X : float
        HF energy in X basis set
    corr_X : float
        Correlation energy in X basis set
    hf_Y : float
        HF energy in Y basis set where Y = X+1 cardinal zeta number
    corr_Y : float
        Correlation energy in Y basis set
    X : int
        Cardinal zeta number of X basis set
    Y : int
        Cardinal zeta number of Y basis set
    family : str
        Basis set family. Options are 'cc', 'def2', 'acc', and 'mixcc'. Where cc is for non-augmented correlation consistent basis sets, def2 is for def2 basis sets, acc is for augmented correlation consistent basis sets while mixcc is for mixed augmented + non-augmented correlation consistent basis sets
    convert_Hartree : bool
        If True, convert energies to Hartree
    shift : float
        Energy shift to apply to the CBS energy
    output : bool
        If True, print CBS energies

    Returns
    -------
    hf_cbs : float
        HF CBS energy
    corr_cbs : float
        Correlation CBS energy
    tot_cbs : float
        Total CBS energy
    """

    # Dictionary of alpha parameters followed by beta parameters in CBS extrapoation. Refer to: Neese, F.; Valeev, E. F. Revisiting the Atomic Natural Orbital Approach for Basis Sets: Robust Systematic Basis Sets for Explicitly Correlated and Conventional Correlated Ab Initio Methods. J. Chem. Theory Comput. 2011, 7 (1), 33–43. https://doi.org/10.1021/ct100396y.
    alpha_dict = {
        "def2_2_3": 10.39,
        "def2_3_4": 7.88,
        "cc_2_3": 4.42,
        "cc_3_4": 5.46,
        "cc_4_5": 5.46,
        "acc_2_3": 4.30,
        "acc_3_4": 5.79,
        "acc_4_5": 5.79,
        "mixcc_2_3": 4.36,
        "mixcc_3_4": 5.625,
        "mixcc_4_5": 5.625,
    }

    beta_dict = {
        "def2_2_3": 2.40,
        "def2_3_4": 2.97,
        "cc_2_3": 2.46,
        "cc_3_4": 3.05,
        "cc_4_5": 3.05,
        "acc_2_3": 2.51,
        "acc_3_4": 3.05,
        "acc_4_5": 3.05,
        "mixcc_2_3": 2.485,
        "mixcc_3_4": 3.05,
        "mixcc_4_5": 3.05,
    }

    # Check if X and Y are consecutive cardinal zeta numbers
    if Y != X + 1:
        print("Y does not equal X+1")

    # Check if basis set family is valid
    if family != "cc" and family != "def2" and family != "acc" and family != "mixcc":
        print("Wrong basis set family stated")

    # Get the corresponding alpha and beta parameters depending on the basis set family
    alpha = alpha_dict["{0}_{1}_{2}".format(family, X, Y)]
    beta = beta_dict["{0}_{1}_{2}".format(family, X, Y)]

    # Perform CBS extrapolation for HF and correlation components
    hf_cbs = hf_X - np.exp(-alpha * np.sqrt(X)) * (hf_Y - hf_X) / (
        np.exp(-alpha * np.sqrt(Y)) - np.exp(-alpha * np.sqrt(X))
    )
    corr_cbs = (X ** (beta) * corr_X - Y ** (beta) * corr_Y) / (
        X ** (beta) - Y ** (beta)
    )

    # Convert energies from Hartree to eV if convert_Hartree is True
    if convert_Hartree == True:
        if output == True:
            print(
                "CBS({0}/{1}) HF: {2:.9f} Corr: {3:.9f} Tot: {4:.9f}".format(
                    X,
                    Y,
                    hf_cbs * Hartree + shift,
                    corr_cbs * Hartree,
                    (hf_cbs + corr_cbs) * Hartree + shift,
                )
            )
        return (
            hf_cbs * Hartree + shift,
            corr_cbs * Hartree,
            (hf_cbs + corr_cbs) * Hartree,
        )
    else:
        if output == True:
            print(
                "CBS({0}/{1})  HF: {2:.9f} Corr: {3:.9f} Tot: {4:.9f}".format(
                    X, Y, hf_cbs + shift, corr_cbs, (hf_cbs + corr_cbs) + shift
                )
            )
        return hf_cbs + shift, corr_cbs, (hf_cbs + corr_cbs) + shift
    
def convert_df_to_latex_input(
    df,
    start_input = '\\begin{table}\n',
    end_input = '\n\\end{table}',
    label = "tab:default",
    caption = "This is a table",
    replace_input = {},
    df_latex_skip = 0,
    adjustbox = 0,
    scalebox = False,
    multiindex_sep = "",
    filename = "./table.tex",
    index = True,
    column_format = None,
    center = False,
    rotate_column_header = False,
    output_str = False,
    float_fmt = None,
    longtable = False,
    longtable_continue_caption = "(continued)",
):
    """
    Convert a pandas DataFrame to LaTeX and optionally emit a full longtable environment.

    Args (new):
      longtable: bool. If True, produce a longtable instead of a table/ tabular snippet.
      longtable_continue_caption: text used for the continued caption in the second chunk.

    All other args preserved / behave as before (with slight adjustments to caption placement
    when longtable=True).
    """

    # default column_format if not provided
    if column_format is None:
        column_format = "l" + "r" * len(df.columns)

    # label + caption pieces
    if label != "":
        label_input = r"\label{" + label + r"}"
    else:
        label_input = ""
    caption_input = r"\caption{" + label_input + caption +  "}"

    # optionally rotate column headers
    if rotate_column_header:
        df = df.copy()
        df.columns = [r'\rotatebox{90}{' + str(col) + '}' for col in df.columns]

    # produce latex from pandas (no external caption yet)
    with pd.option_context("max_colwidth", 1000):
        if float_fmt is not None:
            df_latex_input = df.to_latex(
                escape=False,
                column_format=column_format,
                multicolumn_format='c',
                multicolumn=True,
                index=index,
                float_format=float_fmt
            )
        else:
            df_latex_input = df.to_latex(
                escape=False,
                column_format=column_format,
                multicolumn_format='c',
                multicolumn=True,
                index=index
            )

    # apply simple replacements
    for key in replace_input:
        df_latex_input = df_latex_input.replace(key, replace_input[key])

    # optionally skip top/bottom lines of pandas output (same as before)
    df_latex_input_lines = df_latex_input.splitlines()[df_latex_skip:]

    # find top rule index (as before)
    toprule_index = [i for i, line in enumerate(df_latex_input_lines) if "toprule" in line][0]
    # add multiindex_sep to the header row if requested (keeps your previous behavior)
    df_latex_input_lines[toprule_index+1] = df_latex_input_lines[toprule_index+1] + ' ' + multiindex_sep

    # ===== LONGTABLE handling =====
    if longtable:
        # Replace \begin{tabular}{...} with \begin{longtable}{...}
        joined = '\n'.join(df_latex_input_lines)

        # Replace \begin{tabular}{...} with \begin{longtable}{...}
        joined = re.sub(
            r'\\begin\{tabular\}\{([^\}]*)\}',
            lambda m: r'\begin{longtable}{' + m.group(1) + r'}',
            joined,
            count=1
        )

        # Insert caption right after \begin{longtable}{...}
        joined = re.sub(
            r'(\\begin\{longtable\}\{[^\}]*\})',
            lambda m: m.group(1) + "\n" + caption_input + r" \\",
            joined,
            count=1
        )

        # split back to lines to find header row position (toprule index may have shifted)
        lines = joined.splitlines()

        # recompute top rule and header indices in the modified lines
        try:
            toprule_index2 = [i for i, line in enumerate(lines) if "toprule" in line][0]
            header_idx = toprule_index2 + 1  # header line (column names)
        except IndexError:
            # fallback: place longtable markers near beginning if we can't find top rule
            header_idx = 2

        # number of columns for continued footer (include index column if present)
        n_cols = len(df.columns) + (1 if index else 0)

        # construct the standard longtable continuation blocks (you can customize if needed)
        longtable_blocks = [
            r'\endfirsthead',
            '',
            r'\caption[]{' + longtable_continue_caption + r'} \\',
            r'\endhead',
            '',
            r'\multicolumn{' + str(n_cols) + r'}{r}{{Continued on next page}} \\',
            r'\endfoot',
            '',
            r'\bottomrule',
            r'\endlastfoot',
            ''
        ]

        # insert longtable_blocks into lines after the header row but before the midrule/data
        insert_pos = header_idx + 1
        # guard: don't overflow
        if insert_pos > len(lines):
            insert_pos = len(lines)
        lines[insert_pos:insert_pos] = longtable_blocks

        # Now the pandas produced \end{tabular} still exists; replace it with \end{longtable}
        for i, ln in enumerate(lines):
            if r'\end{tabular}' in ln:
                lines[i] = ln.replace(r'\end{tabular}', r'\end{longtable}')
                break

        df_latex_input = '\n'.join(lines)

        # When using longtable, we won't wrap the output with start_input/ end_input table env
        # so the output_str mode should return df_latex_input directly.
        if output_str:
            return df_latex_input
        else:
            with open(filename, "w") as f:
                f.write(df_latex_input)
            return None

    # ===== non-longtable (original path, slightly refactored) =====
    df_latex_input = '\n'.join(df_latex_input_lines)
    end_adjustbox = False

    if output_str:
        latex_string = ""
        latex_string += start_input + "\n"
        latex_string += caption_input + "\n"
        if center == True and adjustbox == 0:
            latex_string += r"\begin{adjustbox}{center}" + "\n"
            end_adjustbox = True
        elif adjustbox > 0 and center == False:
            latex_string += r"\begin{adjustbox}{max width=" + f"{adjustbox}" + r"\textwidth}" + "\n"
            end_adjustbox = True
        elif adjustbox > 0 and center == True:
            latex_string += r"\begin{adjustbox}{center,max width=" + f"{adjustbox}" + r"\textwidth}" + "\n"
            end_adjustbox = True
        if scalebox:
            latex_string += r"\begin{adjustbox}{scale=" + f"{scalebox}" + "}" + "\n"
            end_adjustbox = True
        latex_string += df_latex_input
        if end_adjustbox:
            latex_string += "\n\\end{adjustbox}"
        latex_string += "\n" + end_input
        return latex_string

    else:
        with open(filename, "w") as f:
            f.write(start_input + "\n")
            f.write(caption_input + "\n")
            if center == True and adjustbox == 0:
                f.write(r"\begin{adjustbox}{center}" + "\n")
                end_adjustbox = True
            elif adjustbox > 0 and center == False:
                f.write(r"\begin{adjustbox}{max width=" + f"{adjustbox}" + r"\textwidth}" + "\n")
                end_adjustbox = True
            elif adjustbox > 0 and center == True:
                f.write(r"\begin{adjustbox}{center,max width=" + f"{adjustbox}" + r"\textwidth}" + "\n")
                end_adjustbox = True
            if scalebox:
                f.write(r"\begin{adjustbox}{scale=" + f"{scalebox}" + "}" + "\n")
                end_adjustbox = True
            f.write(df_latex_input)
            if end_adjustbox:
                f.write("\n\\end{adjustbox}")
            f.write("\n" + end_input)
        return None