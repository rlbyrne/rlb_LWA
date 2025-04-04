import subprocess
import shlex
import os

obsids_list = [
    "density1.5_gleam",
    "density1.5_gsm08",
    "density1.75_gleam",
    "density1.75_gsm08",
    "density2.0_gleam",
    "density2.0_gsm08",
    "density3.0_gleam",
    "density3.0_gsm08",
    "hexa_gleam",
    "hexa_gsm08",
    "pos_error1e-3_gleam",
    "pos_error1e-3_gsm08",
    "random_gleam",
    "random_gsm08",
]
versions_list = ["rlb_vincent_sims_Apr2025"]
uvfits_path = "/lustre/rbyrne/vincent_sims"
outdir = "/lustre/rbyrne/vincent_sims"
run_fhd = True
run_eppsilon = True

# Define wrappers
fhd_versions_script = "fhd_versions_calim"
eppsilon_script = "ps_single_obs_wrapper"

# Set eppsilon options
refresh_ps = 1
uvf_input = 1
no_evenodd = 1  # Use this option if only one time step is present
xx_only = 1

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
