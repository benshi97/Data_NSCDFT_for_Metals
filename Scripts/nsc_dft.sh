#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_vasp_steps.sh "<runner_command>" <mode>
#
# Modes:
#   hbeef   -> run steps 01–04 only
#   dhbeef  -> run all steps 01–08
#
# Inputs:
#   All input files are expected under ./inputs/ :
#     - INCAR_*
#     - KPOINTS_DFT, KPOINTS_RPA (and any other KPOINTS_* you keep)
#     - POTCAR_DFT,  POTCAR_RPA  (and any other POTCAR_* you keep)
#     - POSCAR
#     - vdw_kernel.bindat

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<runner_command>\" <mode: hbeef|dhbeef>"
  exit 1
fi

RUNNER="$1"
MODE="$2"
INPUTS_DIR="inputs"

if [[ "$MODE" != "hbeef" && "$MODE" != "dhbeef" ]]; then
  echo "Error: mode must be 'hbeef' or 'dhbeef'"
  exit 1
fi

[[ -d "$INPUTS_DIR" ]] || { echo "Error: missing inputs directory: $INPUTS_DIR" >&2; exit 1; }

run_vasp() {
  # RUNNER may include flags; allow word splitting intentionally.
  # shellcheck disable=SC2086
  $RUNNER vasp_std
}

step_complete() {
  local step="$1"
  local needs_wavecar="$2"  # yes/no
  local outgz="$step/OUTCAR.gz"

  [[ -f "$outgz" ]] || return 1
  zgrep -q "Total CPU time used" "$outgz" || return 1

  if [[ "$needs_wavecar" == "yes" ]]; then
    [[ -f "$step/WAVECAR.gz" ]] || return 1
  fi
  return 0
}

# Copy a gzipped WAVECAR from a previous step into a run dir as WAVECAR (uncompressed)
load_wavecar_into() {
  local from_step="$1"
  local run_dir="$2"
  local src="$from_step/WAVECAR.gz"
  [[ -f "$src" ]] || { echo "Error: expected $src" >&2; exit 1; }
  cp -f "$src" "$run_dir/WAVECAR.gz"
  (cd "$run_dir" && gunzip -f WAVECAR.gz)
}

# After VASP finishes in tmp_run_*/ , move everything into step/ and gzip files.
finalize_step_from_tmp() {
  local step="$1"
  local tmp="$2"

  shopt -s nullglob dotglob
  for p in "$tmp"/*; do
    [[ -e "$p" ]] || continue
    [[ -d "$p" ]] && continue
    mv -f "$p" "$step/"
  done
  shopt -u nullglob dotglob

  shopt -s nullglob
  for f in "$step"/*; do
    [[ -f "$f" ]] || continue
    [[ "$f" == *.gz ]] && continue
    gzip -f "$f"
  done
  shopt -u nullglob

  rm -rf "$tmp"
}

# Run one step inside step/tmp_run_<timestamp>/
#
# Args:
#   step                e.g. 07_bands
#   incar_name          e.g. INCAR_07_bands (looked up in inputs/)
#   needs_wavecar_out   yes/no (whether we expect WAVECAR.gz at end)
#   wavecar_in_step     "" or step name providing WAVECAR.gz
#   replace_token       "" or token string to replace in INCAR (e.g., XXX)
#   replace_value       "" or value to inject (e.g., nbands)
#   extra_input_name    "" or filename in inputs/ to copy into tmp_run (e.g., vdw_kernel.bindat)
#   kpoints_name        filename in inputs/ to copy as KPOINTS (required)
#   potcar_name         filename in inputs/ to copy as POTCAR (required)
do_step() {
  local step="$1"
  local incar_name="$2"
  local needs_wavecar_out="$3"
  local wavecar_in_step="${4:-}"
  local replace_token="${5:-}"
  local replace_value="${6:-}"
  local extra_input_name="${7:-}"
  local kpoints_name="$8"
  local potcar_name="$9"

  if step_complete "$step" "$needs_wavecar_out"; then
    echo "Skipping $step (already complete)"
    return 0
  fi

  echo "Running $step"

  mkdir -p "$step"
  local ts tmp
  ts="$(date +%Y%m%d_%H%M%S)"
  tmp="$step/tmp_run_${ts}"
  rm -rf "$tmp"
  mkdir -p "$tmp"

  # Stage inputs from inputs/
  [[ -f "$INPUTS_DIR/POSCAR" ]] || { echo "Error: missing $INPUTS_DIR/POSCAR" >&2; exit 1; }
  [[ -f "$INPUTS_DIR/$incar_name" ]] || { echo "Error: missing $INPUTS_DIR/$incar_name" >&2; exit 1; }
  [[ -f "$INPUTS_DIR/$kpoints_name" ]] || { echo "Error: missing $INPUTS_DIR/$kpoints_name" >&2; exit 1; }
  [[ -f "$INPUTS_DIR/$potcar_name" ]] || { echo "Error: missing $INPUTS_DIR/$potcar_name" >&2; exit 1; }

  cp -f "$INPUTS_DIR/POSCAR" "$tmp/POSCAR"
  cp -f "$INPUTS_DIR/$potcar_name" "$tmp/POTCAR"
  cp -f "$INPUTS_DIR/$kpoints_name" "$tmp/KPOINTS"
  cp -f "$INPUTS_DIR/$incar_name" "$tmp/INCAR"

  if [[ -n "$extra_input_name" ]]; then
    [[ -f "$INPUTS_DIR/$extra_input_name" ]] || { echo "Error: missing $INPUTS_DIR/$extra_input_name" >&2; exit 1; }
    cp -f "$INPUTS_DIR/$extra_input_name" "$tmp/"
  fi

  # Inject inside tmp_run for steps that need it
  if [[ -n "$replace_token" ]]; then
    sed -i "s/${replace_token}/${replace_value}/g" "$tmp/INCAR"
  fi

  if [[ -n "$wavecar_in_step" ]]; then
    load_wavecar_into "$wavecar_in_step" "$tmp"
  fi

  # Run
  (cd "$tmp" && run_vasp)

  # Validate
  [[ -f "$tmp/OUTCAR" ]] || { echo "Error: OUTCAR not produced for $step" >&2; exit 1; }

  # Finalize
  finalize_step_from_tmp "$step" "$tmp"

  if [[ "$needs_wavecar_out" == "yes" ]]; then
    [[ -f "$step/WAVECAR.gz" ]] || { echo "Error: $step expected WAVECAR.gz but it was not produced" >&2; exit 1; }
  fi
}

# hBEEF-vdW analysis function
analyze_hbeefvdw_flow() {

  local calc_dir="."
  local h_x_frac="0.175"

  get_energy() {
    local file="$1"
    [[ -f "$file" ]] || { echo "Error: missing $file" >&2; exit 1; }
    zgrep "energy  without entropy=" "$file" | tail -n1 | awk '{print $NF}'
  }

  echo "Analyzing hBEEF-vdW flow (sigma->0 energies)"
  echo "---------------------------------------------"

  beef_xc_vdw=$(get_energy "$calc_dir/01_beefxc_vdw/OUTCAR.gz")
  beef_xc=$(get_energy "$calc_dir/02_beefxc/OUTCAR.gz")
  beef_x=$(get_energy "$calc_dir/03_beefx/OUTCAR.gz")
  exx=$(get_energy "$calc_dir/04_exx/OUTCAR.gz")

  nlc=$(echo "$beef_xc_vdw - $beef_xc" | bc -l)
  beefc=$(echo "$beef_xc - $beef_x" | bc -l)
  beefx="$beef_x"

  hbeefvdw=$(echo "$h_x_frac*$exx + (1-$h_x_frac)*$beefx + $beefc + $nlc" | bc -l)

  printf "%-18s %20.10f\n" "beef_xc_vdw:" "$beef_xc_vdw"
  printf "%-18s %20.10f\n" "beef_xc:"     "$beef_xc"
  printf "%-18s %20.10f\n" "beef_x:"      "$beef_x"
  printf "%-18s %20.10f\n" "exx:"         "$exx"
  echo "---------------------------------------------"
  printf "%-18s %20.10f\n" "hbeef_vdw:"   "$hbeefvdw"
}

# ---------------------------
# Steps 01–04 (DFT KPOINTS/POTCAR)
# ---------------------------
do_step "01_beefxc_vdw" "INCAR_01_beefxc_vdw" "yes" "" "" "" "vdw_kernel.bindat" "KPOINTS_DFT" "POTCAR_DFT"
do_step "02_beefxc"     "INCAR_02_beefxc"     "no"  "01_beefxc_vdw" "" "" ""      "KPOINTS_DFT" "POTCAR_DFT"
do_step "03_beefx"      "INCAR_03_beefx"      "no"  "01_beefxc_vdw" "" "" ""      "KPOINTS_DFT" "POTCAR_DFT"
do_step "04_exx"        "INCAR_04_exx"        "no"  "01_beefxc_vdw" "" "" ""      "KPOINTS_DFT" "POTCAR_DFT"

if [[ "$MODE" == "hbeef" ]]; then
    analyze_hbeefvdw_flow
    exit 0
fi

analyze_dhbeefvdw_flow() {
  local calc_dir="."
  local h_x_frac="0.175"
  local dh_x_frac="0.25"
  local dh_c_frac="0.15"

  # Standard VASP electronic energy: use energy(sigma->0)
  get_sigma0_energy() {
    local file="$1"
    [[ -f "$file" ]] || { echo "Error: missing $file" >&2; exit 1; }
    zgrep "energy  without entropy=" "$file" | tail -n1 | awk '{print $NF}'
  }

  # RPA correlation energy: from
  #   converged value                           -1.7494120986       -2.0468542330
  # take the 3rd column (i.e., first numeric value)
  get_rpac_energy() {
    local file="$1"
    [[ -f "$file" ]] || { echo "Error: missing $file" >&2; exit 1; }
    zgrep -E "^[[:space:]]*converged value" "$file" | tail -n1 | awk '{print $3}'
  }

  echo "Analyzing dhBEEF-vdW flow (sigma->0 + RPA converged value col3)"
  echo "----------------------------------------------------------------"

  # Steps 1-4: BEEF components
  local beef_xc_vdw beef_xc beef_x exx
  beef_xc_vdw=$(get_sigma0_energy "$calc_dir/01_beefxc_vdw/OUTCAR.gz")
  beef_xc=$(get_sigma0_energy     "$calc_dir/02_beefxc/OUTCAR.gz")
  beef_x=$(get_sigma0_energy      "$calc_dir/03_beefx/OUTCAR.gz")
  exx=$(get_sigma0_energy         "$calc_dir/04_exx/OUTCAR.gz")

  # hBEEF-vdW pieces
  local nlc beefc beefx exx_energy hbeef_vdw
  nlc=$(echo "$beef_xc_vdw - $beef_xc" | bc -l)
  beefc=$(echo "$beef_xc - $beef_x" | bc -l)
  beefx="$beef_x"
  exx_energy="$exx"
  hbeef_vdw=$(echo "$h_x_frac*$exx_energy + (1-$h_x_frac)*$beefx + $beefc + $nlc" | bc -l)

  # Steps 5-6: PBE + PBE_EXX for RPA
  local pbe pbe_exx
  pbe=$(get_sigma0_energy     "$calc_dir/05_pbe_rpa/OUTCAR.gz")
  pbe_exx=$(get_sigma0_energy "$calc_dir/06_pbe_exx/OUTCAR.gz")

  # Step 8: RPA correlation energy
  local rpac rpa
  rpac=$(get_rpac_energy "$calc_dir/08_rpa/OUTCAR.gz")
  rpa=$(echo "$pbe_exx + $rpac" | bc -l)

  # dhBEEF-vdW energy
  local dhbeef_vdw
  dhbeef_vdw=$(echo "$dh_x_frac*$exx_energy + (1-$dh_x_frac)*$beefx + $dh_c_frac*$rpac + (1-$dh_c_frac)*$beefc + $nlc" | bc -l)

  # Print summary
  printf "%-18s %20.10f\n" "beef_xc_vdw:" "$beef_xc_vdw"
  printf "%-18s %20.10f\n" "beef_xc:"     "$beef_xc"
  printf "%-18s %20.10f\n" "beef_x:"      "$beef_x"
  printf "%-18s %20.10f\n" "exx:"         "$exx_energy"
  printf "%-18s %20.10f\n" "hbeef_vdw:"   "$hbeef_vdw"
  echo "----------------------------------------------------------------"
  printf "%-18s %20.10f\n" "pbe:"         "$pbe"
  printf "%-18s %20.10f\n" "pbe_exx:"     "$pbe_exx"
  printf "%-18s %20.10f\n" "rpac:"        "$rpac"
  printf "%-18s %20.10f\n" "rpa:"         "$rpa"
  echo "----------------------------------------------------------------"
  printf "%-18s %20.10f\n" "dhbeef_vdw:"  "$dhbeef_vdw"
}


# ---------------------------
# Steps 05–08 (RPA KPOINTS/POTCAR)
# ---------------------------

# Step 05 produces WAVECAR and defines nbands
if step_complete "05_pbe_rpa" "yes"; then
  echo "Skipping 05_pbe_rpa (already complete)"
  nbands=$(zgrep -m1 "maximum number of plane-waves" 05_pbe_rpa/OUTCAR.gz | awk '{print $5}')
else
  do_step "05_pbe_rpa" "INCAR_05_pbe_rpa" "yes" "" "" "" "" "KPOINTS_RPA" "POTCAR_RPA"
  nbands=$(zgrep -m1 "maximum number of plane-waves" 05_pbe_rpa/OUTCAR.gz | awk '{print $5}')
fi

# Step 06 loads WAVECAR from 05
do_step "06_pbe_exx" "INCAR_06_pbe_exx" "no" "05_pbe_rpa" "" "" "" "KPOINTS_RPA" "POTCAR_RPA"

# Step 07 loads WAVECAR from 05, inject nbands, produces WAVECAR
do_step "07_bands" "INCAR_07_bands" "yes" "05_pbe_rpa" "XXX" "$nbands" "" "KPOINTS_RPA" "POTCAR_RPA"

# Step 08 loads WAVECAR from 07, inject nbands
do_step "08_rpa" "INCAR_08_rpa" "no" "07_bands" "XXX" "$nbands" "" "KPOINTS_RPA" "POTCAR_RPA"

if [[ "$MODE" == "dhbeef" ]]; then
  analyze_dhbeefvdw_flow
fi
