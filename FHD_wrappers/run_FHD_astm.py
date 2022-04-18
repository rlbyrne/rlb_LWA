import subprocess
import shlex
import os

obsids_list = [
    "20220307_175923_61MHz_uncalib"
]
versions_list = ["rlb_LWA_imaging_optimal_Apr2022"]
uvfits_path = "/lustre/rbyrne/LWA_data_20220307/uvfits"
outdir = "/lustre/rbyrne/fhd_outputs"
run_eppsilon = False

# Define wrappers
fhd_versions_script = "fhd_versions_astm"
eppsilon_script = "ps_single_obs_wrapper"

# Set eppsilon options
refresh_ps = 0
uvf_input = 0

for version in versions_list:
    # Create directories
    if not os.path.isdir(f"{outdir}/fhd_{version}"):
        os.mkdir(f"{outdir}/fhd_{version}")
    if not os.path.isdir(f"{outdir}/fhd_{version}/logs"):
        os.mkdir(f"{outdir}/fhd_{version}/logs")

    for obsid in obsids_list:
        # Run FHD
        with open(f"{outdir}/fhd_{version}/logs/{obsid}_fhd_stdout.txt", "wb") as out, open(
            f"{outdir}/fhd_{version}/logs/{obsid}_fhd_stderr.txt", "wb"
        ) as err:
            process = subprocess.Popen(
                shlex.split(
                    f"/opt/astro/devel/harris/idl87/bin/idl -e {fhd_versions_script} -args {outdir} {version} {uvfits_path}/{obsid}.uvfits"
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
                        f"/opt/astro/devel/harris/idl87/bin/idl -e {eppsilon_script} -args {obsid} {outdir} {version} {refresh_ps} {uvf_input}"
                    ),
                    stdout=out,
                    stderr=err,
                )
            stdout, stderr = process.communicate()
