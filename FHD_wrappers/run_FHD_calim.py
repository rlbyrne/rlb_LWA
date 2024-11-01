import subprocess
import shlex
import os

obsids_list = ["41-20240303_093000-093151_41-82MHz_calibrated_core"]
versions_list = ["rlb_process_LWA_Sept2024"]
uvfits_path = "/lustre/rbyrne/2024-03-03"
outdir = "/lustre/rbyrne/fhd_outputs"
run_fhd = True
run_eppsilon = True

# Define wrappers
fhd_versions_script = "fhd_versions_calim"
eppsilon_script = "ps_single_obs_wrapper"

# Set eppsilon options
refresh_ps = 1
uvf_input = 1
no_evenodd = 0  # Use this option if only one time step is present
xx_only = 0

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
                        f"/opt/devel/rbyrne/harris/idl88/bin/idl -e {fhd_versions_script} -args {outdir} {version} {uvfits_path}/{obsid}.uvfits"
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
                        f"/opt/devel/rbyrne/harris/idl88/bin/idl -e {eppsilon_script} -args {obsid} {outdir} {version} {refresh_ps} {uvf_input} {no_evenodd} {xx_only}"
                    ),
                    stdout=out,
                    stderr=err,
                )
            stdout, stderr = process.communicate()
