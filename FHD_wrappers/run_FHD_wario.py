import subprocess
import shlex
import os

obsids_list = [
    'OVRO-LWA_1h_sim_center_time',
    'OVRO-LWA_1h_sim_phased',
    'OVRO-LWA_1h_sim_unphased',
    'OVRO-LWA_100ms_sim_center_time',
    'OVRO-LWA_100ms_sim_phased',
    'OVRO-LWA_100ms_sim_unphased'
]
versions_list = [
    'rlb_LWA_phasing_sim_Feb2021'
]
uvfits_path = '/safepool/rbyrne/LWA_pyuvsim_simulations'
outdir = '/safepool/rbyrne/LWA_pyuvsim_simulations'
run_eppsilon = False

# Define wrappers
fhd_versions_script = 'fhd_versions_wario'
eppsilon_script = 'ps_single_obs_wrapper'

# Set eppsilon options
refresh_ps = 0
uvf_input = 0

for version in versions_list:
    # Create directories
    if not os.path.isdir(f'{outdir}/fhd_{version}'):
        os.mkdir(f'{outdir}/fhd_{version}')
    if not os.path.isdir(f'{outdir}/fhd_{version}/logs'):
        os.mkdir(f'{outdir}/fhd_{version}/logs')

    for obsid in obsids_list:
        # Run FHD
        with open(
            f'{outdir}/fhd_{version}/logs/{obsid}_fhd_stdout.txt', 'wb'
        ) as out, open(
            f'{outdir}/fhd_{version}/logs/{obsid}_fhd_stderr.txt', 'wb'
        ) as err:
            process = subprocess.Popen(shlex.split(
                f'/opt/idl/idl88/bin/idl -e {fhd_versions_script} -args {outdir} {version} {uvfits_path}/{obsid}.uvfits'
            ), stdout=out, stderr=err)
        stdout, stderr = process.communicate()

        # Run eppsilon
        if run_eppsilon:
            with open(
                f'{outdir}/fhd_{version}/logs/{obsid}_eppsilon_stdout.txt', 'wb'
            ) as out, open(
                f'{outdir}/fhd_{version}/logs/{obsid}_eppsilon_stderr.txt', 'wb'
            ) as err:
                process = subprocess.Popen(shlex.split(
                    f'/opt/astro/devel/harris/idl87/bin/idl -e {eppsilon_script} -args {obsid} {outdir} {version} {refresh_ps} {uvf_input}'
                ), stdout=out, stderr=err)
            stdout, stderr = process.communicate()
