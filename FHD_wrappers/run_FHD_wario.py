import subprocess
import shlex
import os

obsids_list = [
    #"sim_uv_spacing_10",
    "sim_uv_spacing_5",
    "sim_uv_spacing_1",
    #"sim_uv_spacing_0.5"
]
versions_list = ["rlb_process_uv_density_sims_Feb2023"]
uvfits_path = "/safepool/rbyrne/uv_density_simulations"
outdir = "/safepool/rbyrne/fhd_outputs"
run_fhd = False
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
