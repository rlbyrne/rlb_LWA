import subprocess
import shlex
import os

obsids_list = [
    "20230819_093023_73MHz_calibrated",
    "20230819_093023_73MHz_data_minus_cyg_cas",
    "20230819_093023_73MHz_data_minus_deGasperin_cyg_cas",
    "20230819_093023_73MHz_data_minus_deGasperin_sources",
    "20230819_093023_73MHz_data_minus_VLSS"
]
versions_list = [
    "rlb_LWA_image_Dec2023",
]
uvfits_path = "/safepool/rbyrne/pyuvsim_sims_Dec2023"
outdir = "/safepool/rbyrne/fhd_outputs"
run_fhd = True
run_eppsilon = True

# Define wrappers
fhd_versions_script = "fhd_versions_wario"
eppsilon_script = "ps_single_obs_wrapper"

# Set eppsilon options
refresh_ps = 1
uvf_input = 1
no_evenodd = 1  # Use this option if only one time step is present

for version in versions_list:
    # Create directories
    if not os.path.isdir(f"{outdir}/fhd_{version}"):
        os.mkdir(f"{outdir}/fhd_{version}")
    if not os.path.isdir(f"{outdir}/fhd_{version}/logs"):
        os.mkdir(f"{outdir}/fhd_{version}/logs")

    for obsid in obsids_list:
        # Run FHD
        if run_fhd:
            with open(
                f"{outdir}/fhd_{version}/logs/{obsid}_fhd_stdout.txt", "wb"
            ) as out, open(
                f"{outdir}/fhd_{version}/logs/{obsid}_fhd_stderr.txt", "wb"
            ) as err:
                process = subprocess.Popen(
                    shlex.split(
                        f"/opt/idl/idl88/bin/idl -e {fhd_versions_script} -args {outdir} {version} {uvfits_path}/{obsid}.uvfits"
                    ),
                    stdout=out,
                    stderr=err,
                )
            stdout, stderr = process.communicate()

        # Run eppsilon
        if run_eppsilon:
            with open(
                f"{outdir}/fhd_{version}/logs/{obsid}_eppsilon_stdout.txt", "wb"
            ) as out, open(
                f"{outdir}/fhd_{version}/logs/{obsid}_eppsilon_stderr.txt", "wb"
            ) as err:
                process = subprocess.Popen(
                    shlex.split(
                        f"/opt/idl/idl88/bin/idl -e {eppsilon_script} -args {obsid} {outdir} {version} {refresh_ps} {uvf_input} {no_evenodd}"
                    ),
                    stdout=out,
                    stderr=err,
                )
            stdout, stderr = process.communicate()
