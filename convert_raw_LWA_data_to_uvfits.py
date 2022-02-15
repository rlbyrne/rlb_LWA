import pyuvdata
import os
import numpy as np
import subprocess
import shlex

data_path = "/lustre/rbyrne/LWA_data_20220210"
freq = "70MHz"
start_time_stamp = 191447
end_time_stamp = 194824
outfile_name = f"/lustre/rbyrne/LWA_data_02102022/20220210_{freq}_combined.uvfits"

filenames = os.listdir(data_path)
use_files = []
for file in filenames:
    file_split = file.split("_")
    if file_split[2] == "{}.ms.tar".format(freq):
        if int(file_split[1]) >= start_time_stamp and int(file_split[1]) <= end_time_stamp:
            use_files.append(file)

for file_ind, file in enumerate(use_files):
    subprocess.call(shlex.split(f"tar -xvf {data_path}/{file} -C {data_path}"))
    file_split = file.split(".")
    uv_new = pyuvdata.UVData()
    uv_new.read_ms(f"{data_path}/{file_split[0]}.ms")
    #uv_new.phase_type = "drift"
    if file_ind == 0:
        uv = uv_new
    else:
        uv = uv + uv_new
    subprocess.call(shlex.split(f"rm -r {data_path}/{file_split[0]}.ms"))
print(uv.check())

uv.instrument = 'OVRO-LWA'
uv.telescope_name = 'OVRO-LWA'
uv.set_telescope_params()
print('Saving file to {}'.format(outfile_name))
uv.write_uvfits(outfile_name, spoof_nonessential=True)
