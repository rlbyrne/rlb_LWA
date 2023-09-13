import subprocess
import shlex
import os

obsids_list = [
    "120952high.peel"
]
versions_list = [
    "rlb_image_LWA_modified_kernel_test_branch_Sept2023",
]
uvfits_path = "/safepool/rbyrne/lwa_data"
outdir = "/safepool/rbyrne/fhd_outputs"
run_fhd = True
run_eppsilon = False

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
