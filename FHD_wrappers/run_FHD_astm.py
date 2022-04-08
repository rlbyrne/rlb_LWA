import subprocess
import shlex
import os

obsids_list = [
    "20220304_181704_15MHz",
    "20220304_181704_20MHz",
    "20220304_181704_24MHz",
    "20220304_181704_29MHz",
    "20220304_181704_34MHz",
    "20220304_181704_38MHz",
    "20220304_181704_43MHz",
    "20220304_181704_47MHz",
    "20220304_181704_52MHz",
    "20220304_181704_57MHz",
    "20220304_181704_61MHz",
    "20220304_181704_66MHz",
    "20220304_181704_70MHz",
    "20220304_181704_75MHz",
    "20220304_181704_80MHz",
    "20220304_181704_84MHz"
]
versions_list = ["rlb_LWA_imaging_Apr2022"]
uvfits_path = "/lustre/rbyrne/LWA_data_20220304/uvfits_ssins_flagged"
outdir = "/lustre/rbyrne/fhd_outputs"
run_eppsilon = True

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
