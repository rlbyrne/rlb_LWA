import subprocess
import shlex
import os

obsid = '2019-11-21T23:00:08'
version = 'rlb_test_LWA_increase_uv_resolution'
uvfits_path = '/lustre/rbyrne'
outdir = '/lustre/rbyrne/fhd_outputs'
run_eppsilon = False

# Define wrappers
fhd_versions_script = 'fhd_versions_astm'
eppsilon_script = 'ps_single_obs_astm'

# Set eppsilon options
refresh_ps = 0
uvf_input = 0

# Create directories
if not os.path.isdir(f'{outdir}/fhd_{version}'):
    os.mkdir(f'{outdir}/fhd_{version}')
if not os.path.isdir(f'{outdir}/fhd_{version}/logs'):
    os.mkdir(f'{outdir}/fhd_{version}/logs')

# Run FHD
with open(
    f'{outdir}/fhd_{version}/logs/{obsid}_fhd_stdout.txt', 'wb'
) as out, open(
    f'{outdir}/fhd_{version}/logs/{obsid}_fhd_stderr.txt', 'wb'
) as err:
    process = subprocess.Popen(shlex.split(
        f'/opt/astro/devel/harris/idl87/bin/idl -e {fhd_versions_script} -args {outdir} {version} {uvfits_path}/{obsid}.uvfits'
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
