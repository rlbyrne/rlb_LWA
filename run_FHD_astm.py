import subprocess
import shlex

version = 'rlb_test_run_Aug2021'
uvfits_path = '/lustre/rbyrne/MWA_data/1061316296.uvfits'
outdir = '/lustre/rbyrne/fhd_outputs'
process = subprocess.Popen(shlex.split(f'/opt/astro/devel/harris/idl87/bin/idl -e fhd_versions_astm -args {outdir} {version} {uvfits_path}'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
