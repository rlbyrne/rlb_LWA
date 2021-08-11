import subprocess
import shlex

version = 'rlb_test_run_Aug2021'
uvfits_path = '/lustre/rbyrne/MWA_data/1061316296.uvfits'
outdir = '/lustre/rbyrne/fhd_outputs'
versions_script = 'fhd_versions_astm'

with open(f'{outdir}/fhd_{version}/stdout.txt', 'wb') as out, open(f'{outdir}/fhd_{version}/stderr.txt', 'wb') as err:
	process = subprocess.Popen(shlex.split(f'/opt/astro/devel/harris/idl87/bin/idl -e {versions_script} -args {outdir} {version} {uvfits_path}'), stdout=out, stderr=err)
stdout, stderr = process.communicate()
