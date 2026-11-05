from LWA_calibrate import *
from calico import calibration_wrappers, cost_function_calculations, caldata
import pyuvdata
import os
import numpy as np
import sys
import time
import pathlib
import shlex
import glob
import subprocess
import pyuvdata

# Assumed to already be imported/defined in your module, as in the original:
#   calibration_pipeline, calibration_wrappers

PROCESSING_PATH = "/fast/rbyrne/pipeline_tmp"
FINAL_PATH = "/lustre/21cmpipe/2026-10-22"
IDL_BIN = "/opt/devel/rbyrne/harris/idl88/bin/idl"


def find_output(relative_pattern, search_roots):
    """
    Look for a file/dir matching a glob pattern (relative to each root).
    Returns the first match found, or None. Outputs may be in the tmp
    processing dir (not yet rsynced) or the final dir (already rsynced),
    so both are searched.
    """
    for root in search_roots:
        matches = sorted(
            glob.glob(os.path.join(root, relative_pattern), recursive=True)
        )
        if matches:
            return matches[0]
    return None


def find_all_outputs(relative_patterns, search_roots):
    """
    Returns a list of matched paths (one per pattern) if *every* pattern has
    a match, otherwise None.
    """
    found = []
    for pattern in relative_patterns:
        match = find_output(pattern, search_roots)
        if match is None:
            return None
        found.append(match)
    return found


def run_idl_logged(command, stdout_path, stderr_path):
    """Run an IDL command, logging stdout/stderr. Returns the return code."""
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        process = subprocess.Popen(shlex.split(command), stdout=out, stderr=err)
        process.communicate()
    return process.returncode


def process_data():

    processing_path = PROCESSING_PATH
    final_path = FINAL_PATH
    search_roots = [processing_path, final_path]
    #pipeline_calibration_path = "/fast/rbyrne/calibration_2026-04-19_17h.B.flagged"
    pipeline_calibration_path = "/lustre/pipeline/calibration/results/2026-10-22/05h/successful/20261022_132848/tables/calibration_2026-10-22_05h.B.flagged"
    flag_antenna_list = [
        "LWA-266B",
        "LWA-269B",
        "LWA-282B",
        "LWA-310B",
        "LWA-341A",
        "LWA-341B",
        "LWA-255B",
        "LWA-263A",
        "LWA-263B",
        "LWA-272A",
        "LWA-272B",
        "LWA-283A",
        "LWA-280A",
        "LWA-280B",
        "LWA-288B",
        "LWA-284A",
        "LWA-292B",
        "LWA-336A",
        "LWA-336B",
        "LWA-335A",
        "LWA-004B",
        "LWA-005B",
        "LWA-040A",
        "LWA-302A",
        "LWA-302B",
        "LWA-013B",
        "LWA-016B",
        "LWA-015A",
        "LWA-015B",
        "LWA-014A",
        "LWA-014B",
        "LWA-025A",
        "LWA-348B",
        "LWA-365A",
        "LWA-365B",
        "LWA-364A",
        "LWA-364B",
        "LWA-086A",
        "LWA-086B",
        "LWA-122A",
        "LWA-334A",
        "LWA-361A",
        "LWA-068B",
        "LWA-069A",
        "LWA-252A",
        "LWA-252B",
        "LWA-289A",
        "LWA-299B",
        "LWA-107A",
        "LWA-110A",
        "LWA-112A",
        "LWA-112B",
        "LWA-111A",
        "LWA-143B",
        "LWA-124A",
        "LWA-159A",
        "LWA-224A",
        "LWA-347A",
        "LWA-167B",
        "LWA-227B",
        "LWA-329A",
        "LWA-234A",
        "LWA-206A",
        "LWA-206B",
        "LWA-210A",
        "LWA-235B",
        "LWA-342B",
        "LWA-214B",
        "LWA-221B",
        "LWA-315A",
    ]
    use_frequencies = [
        "34",
        #"44",
        #"52",
        #"62",
        #"72",
        #"79",
        #"83",
    ]
    date = "2026-10-22"
    fhd_version = "rlb_process_LWA_modified_kernel_Jun2026"

    # Collect input files
    filepaths = []
    for use_freq in use_frequencies:
        for use_hour in os.listdir(
            f"/lustre/pipeline/cosmology/concatenated_data/{use_freq}MHz/{date}"
        ):
            for filename in os.listdir(
                f"/lustre/pipeline/cosmology/concatenated_data/{use_freq}MHz/{date}/{use_hour}"
            ):
                filepaths.append(
                    f"/lustre/pipeline/cosmology/concatenated_data/{use_freq}MHz/{date}/{use_hour}/{filename}"
                )

    for filepath in filepaths:

        filename = os.path.basename(filepath)
        filename_start = filename.split(".")[0]
        new_filepath = f"{processing_path}/{filename}"
        obsid = f"{filename_start}_selfcal_peeled"

        # ------------------------------------------------------------------
        # Expected outputs of each step (relative to processing/final dirs).
        # NOTE: the FHD and eppsilon patterns are my best guess at the usual
        # output layout; adjust them if your outputs are named differently.
        # ------------------------------------------------------------------
        cal_ms_rel = f"{filename_start}_05h_cal_calibrated.ms"
        selfcal_calfits_rel = f"{filename_start}_selfcal.calfits"
        peeled_ms_rel = f"{obsid}.ms"
        uvfits_rel = f"{obsid}.uvfits"
        fhd_output_rels = [
            f"fhd_{fhd_version}/metadata/{obsid}_params.sav",
        ]
        eppsilon_output_rels = [
            f"fhd_{fhd_version}/ps/**/{obsid}*",
        ]

        # Check what already exists
        cal_ms = find_output(cal_ms_rel, search_roots)
        selfcal_calfits = find_output(selfcal_calfits_rel, search_roots)
        peeled_ms = find_output(peeled_ms_rel, search_roots)
        uvfits = find_output(uvfits_rel, search_roots)
        fhd_outputs = find_all_outputs(fhd_output_rels, search_roots)
        eppsilon_outputs = find_all_outputs(eppsilon_output_rels, search_roots)

        if all(
            x is not None
            for x in [
                cal_ms,
                selfcal_calfits,
                peeled_ms,
                uvfits,
                fhd_outputs,
                eppsilon_outputs,
            ]
        ):
            print(f"{filename_start}: all outputs exist, skipping.")
            continue

        os.makedirs(processing_path, exist_ok=True)

        # Raw data is only copied if a step that needs it has to run
        def ensure_input_copied():
            if not os.path.exists(new_filepath):
                subprocess.run(
                    ["cp", "-r", filepath, f"{processing_path}/"], check=True
                )

        try:
            # --------------------------------------------------------------
            # 1. Apply pipeline calibration
            # --------------------------------------------------------------
            if cal_ms is None:
                print(f"{filename_start}: applying pipeline calibration")
                ensure_input_copied()
                calibration_pipeline(
                    new_filepath,
                    output_dir=None,
                    tmp_dir=None,
                    cal_trial_name="17h_cal",
                    apply_cal_path=pipeline_calibration_path,
                    run_aoflagger=True,
                    aoflagger_strategy_file="/lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua",
                    flag_antennas_from_autocorrs=True,
                    #flag_antenna_list=flag_antenna_list,
                    refresh_flags=True,
                    flip_gain_conj=True,
                    plot_gains=False,
                    apply_calibration=True,
                    smooth_cal=False,
                    plot_images=True,
                    peel=False,
                )
                cal_ms = find_output(cal_ms_rel, search_roots)
                if cal_ms is None:
                    raise RuntimeError(
                        f"Pipeline calibration did not produce {cal_ms_rel}"
                    )
            else:
                print(f"{filename_start}: pipeline calibration output exists, skipping")

            # --------------------------------------------------------------
            # 2. Run selfcal
            # --------------------------------------------------------------
            if selfcal_calfits is None:
                print(f"{filename_start}: running selfcal")
                ensure_input_copied()
                data = pyuvdata.UVData()
                data.read(new_filepath, data_column="DATA")
                model = pyuvdata.UVData()
                model.read(cal_ms, data_column="MODEL_DATA")
                uvcal = calibration_wrappers.sky_based_calibration_wrapper(
                    data,
                    model,
                    min_cal_baseline_lambda=10,
                    max_cal_baseline_lambda=125,
                    gains_multiply_model=True,
                    verbose=True,
                    get_crosspol_phase=False,
                    xtol=1e-6,
                    maxiter=200,
                    antenna_flagging_iterations=0,
                    parallel=True,
                    lambda_val=0,
                )
                uvcal.write_calfits(
                    f"{processing_path}/{selfcal_calfits_rel}",
                    clobber=True,
                )
                selfcal_calfits = find_output(selfcal_calfits_rel, search_roots)
                if selfcal_calfits is None:
                    raise RuntimeError(f"Selfcal did not produce {selfcal_calfits_rel}")
            else:
                print(f"{filename_start}: selfcal output exists, skipping")

            # --------------------------------------------------------------
            # 3. Apply selfcal (and peel)
            # --------------------------------------------------------------
            if peeled_ms is None:
                print(f"{filename_start}: applying selfcal and peeling")
                ensure_input_copied()
                calibration_pipeline(
                    new_filepath,
                    output_dir=processing_path,
                    tmp_dir=None,
                    cal_trial_name="selfcal",
                    apply_cal_path=selfcal_calfits,
                    run_aoflagger=False,
                    flag_antennas_from_autocorrs=True,
                    refresh_flags=False,
                    flip_gain_conj=False,
                    plot_gains=False,
                    apply_calibration=True,
                    smooth_cal=True,
                    plot_images=True,
                    peel=True,
                )
                peeled_ms = find_output(peeled_ms_rel, search_roots)
                if peeled_ms is None:
                    raise RuntimeError(
                        f"Selfcal apply/peel did not produce {peeled_ms_rel}"
                    )
            else:
                print(f"{filename_start}: peeled ms exists, skipping")

            # --------------------------------------------------------------
            # 4. Convert to uvfits
            # --------------------------------------------------------------
            if uvfits is None:
                print(f"{filename_start}: writing uvfits")
                uv = pyuvdata.UVData()
                uv.read(peeled_ms)
                uv.write_uvfits(
                    f"{processing_path}/{uvfits_rel}",
                    uvw_double=False,
                )
                uvfits = find_output(uvfits_rel, search_roots)
                if uvfits is None:
                    raise RuntimeError(
                        f"uvfits conversion did not produce {uvfits_rel}"
                    )
            else:
                print(f"{filename_start}: uvfits exists, skipping")

            # --------------------------------------------------------------
            # 5. Run FHD
            # --------------------------------------------------------------
            fhd_versions_script = "fhd_versions_calim"
            if fhd_outputs is None:
                print(f"{filename_start}: running FHD")
                use_outdir = processing_path
                use_uvfits_path = os.path.dirname(uvfits)
                os.makedirs(f"{use_outdir}/fhd_{fhd_version}/logs", exist_ok=True)
                returncode = run_idl_logged(
                    f"{IDL_BIN} -e {fhd_versions_script} -args {use_outdir} {fhd_version} {use_uvfits_path}/{obsid}.uvfits",
                    f"{use_outdir}/fhd_{fhd_version}/logs/{obsid}_fhd_stdout.txt",
                    f"{use_outdir}/fhd_{fhd_version}/logs/{obsid}_fhd_stderr.txt",
                )
                fhd_outputs = find_all_outputs(fhd_output_rels, search_roots)
                if fhd_outputs is None:
                    raise RuntimeError(
                        f"FHD did not produce expected outputs (return code {returncode}); "
                        f"see {use_outdir}/fhd_{fhd_version}/logs/{obsid}_fhd_stderr.txt"
                    )
            else:
                print(f"{filename_start}: FHD outputs exist, skipping")

            # --------------------------------------------------------------
            # 6. Run eppsilon
            # --------------------------------------------------------------
            eppsilon_script = "ps_single_obs_wrapper"
            if eppsilon_outputs is None:
                print(f"{filename_start}: running eppsilon")
                # Run eppsilon from wherever the FHD outputs currently live
                # (tmp dir, or the final dir if they were already rsynced)
                fhd_params_path = find_output(fhd_output_rels[0], search_roots)
                # .../<root>/fhd_<version>/metadata/<obsid>_params.sav -> <root>
                use_outdir = os.path.dirname(
                    os.path.dirname(os.path.dirname(fhd_params_path))
                )
                os.makedirs(f"{use_outdir}/fhd_{fhd_version}/logs", exist_ok=True)

                # Set eppsilon options
                refresh_ps = 1
                uvf_input = 1
                no_evenodd = 0  # Use this option if only one time step is present
                xx_only = 0
                float_colorbar = 1

                returncode = run_idl_logged(
                    f"{IDL_BIN} -e {eppsilon_script} -args {obsid} {use_outdir} {fhd_version} {refresh_ps} {uvf_input} {no_evenodd} {xx_only} {float_colorbar}",
                    f"{use_outdir}/fhd_{fhd_version}/logs/{obsid}_eppsilon_stdout.txt",
                    f"{use_outdir}/fhd_{fhd_version}/logs/{obsid}_eppsilon_stderr.txt",
                )
                eppsilon_outputs = find_all_outputs(eppsilon_output_rels, search_roots)
                if eppsilon_outputs is None:
                    raise RuntimeError(
                        f"eppsilon did not produce expected outputs (return code {returncode}); "
                        f"see {use_outdir}/fhd_{fhd_version}/logs/{obsid}_eppsilon_stderr.txt"
                    )
            else:
                print(f"{filename_start}: eppsilon outputs exist, skipping")

        except Exception as e:
            # Leave partial outputs where they are (no rsync/cleanup) so the
            # next run resumes from the last completed step.
            print(f"{filename_start}: FAILED -- {e}")
            continue

        # ------------------------------------------------------------------
        # Move everything to the final location
        # ------------------------------------------------------------------
        if os.path.isdir(processing_path) and os.listdir(processing_path):
            subprocess.run(
                [
                    "sudo",
                    "rsync",
                    "-a",
                    "--remove-source-files",
                    f"{processing_path}/",
                    final_path,
                ],
                check=True,
            )
            subprocess.run(
                ["find", f"{processing_path}/", "-type", "d", "-empty", "-delete"],
                check=True,
            )


if __name__ == "__main__":
    process_data()
