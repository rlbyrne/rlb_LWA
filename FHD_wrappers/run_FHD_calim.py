import subprocess
import shlex
import os

obsids_list = ["cal46_time11_newcal_cyg_cas"]
versions_list = ["rlb_image_LWA_May2024"]
uvfits_path = "/data03/rbyrne/20231222/test_pyuvsim_modeling"
outdir = "/data03/rbyrne/20231222/fhd_outputs"
run_fhd = True
run_eppsilon = True

# Define wrappers
fhd_versions_script = "fhd_versions_calim"
eppsilon_script = "ps_single_obs_wrapper"

# Set eppsilon options
refresh_ps = 1
uvf_input = 1
no_evenodd = 1  # Use this option if only one time step is present
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
                        f"/opt/idl/idl88/bin/idl -e {eppsilon_script} -args {obsid} {outdir} {version} {refresh_ps} {uvf_input} {no_evenodd} {xx_only}"
                    ),
                    stdout=out,
                    stderr=err,
                )
            stdout, stderr = process.communicate()
