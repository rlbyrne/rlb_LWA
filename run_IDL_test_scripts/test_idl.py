import subprocess
import shlex

process = subprocess.Popen(shlex.split(f'/opt/astro/devel/harris/idl87/bin/idl -e test_idl -args 4 5'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(stderr)
