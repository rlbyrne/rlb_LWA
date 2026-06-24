import subprocess
import shlex
import os
import pathlib
import numpy as np

obsids_list = ["20260419_055641-055832_44MHz_17h_cal_peeled"]
versions_list = ["rlb_process_LWA_modified_kernel_Jun2026"]
uvfits_path = "/fast/rbyrne"
outdir = "/fast/rbyrne/fhd_outputs"
tmp_dir = None
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
float_colorbar = 1

for obsid in obsids_list:

    if tmp_dir is not None:
        os.system(f"cp {uvfits_path}/{obsid}.uvfits {tmp_dir}")
        use_uvfits_path = tmp_dir
        use_outdir = tmp_dir
    else:
        use_uvfits_path = uvfits_path
        use_outdir = outdir

    for version in versions_list:
        # Create directories
        if not os.path.isdir(f"{outdir}/fhd_{version}"):
            pathlib.Path(f"{outdir}/fhd_{version}").mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(f"{outdir}/fhd_{version}/logs"):
            pathlib.Path(f"{outdir}/fhd_{version}/logs").mkdir(parents=True, exist_ok=True)

        try:
            # Run FHD
            if run_fhd:
                with open(
                    f"{outdir}/fhd_{version}/logs/{obsid}_fhd_stdout.txt", "wb"
                ) as out, open(
                    f"{outdir}/fhd_{version}/logs/{obsid}_fhd_stderr.txt", "wb"
                ) as err:
                    process = subprocess.Popen(
                        shlex.split(
                            f"/opt/devel/rbyrne/harris/idl88/bin/idl -e {fhd_versions_script} -args {use_outdir} {version} {use_uvfits_path}/{obsid}.uvfits"
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
                            f"/opt/devel/rbyrne/harris/idl88/bin/idl -e {eppsilon_script} -args {obsid} {use_outdir} {version} {refresh_ps} {uvf_input} {no_evenodd} {xx_only} {float_colorbar}"
                        ),
                        stdout=out,
                        stderr=err,
                    )
                stdout, stderr = process.communicate()
        except:
            pass

        if tmp_dir is not None:
            os.system(f"cp -R {tmp_dir}/fhd_{version}/* {outdir}/fhd_{version}/")
            os.system(f"rm -R {tmp_dir}/fhd_{version}")

    if tmp_dir is not None:
        os.system(f"rm {tmp_dir}/{obsid}.uvfits")

