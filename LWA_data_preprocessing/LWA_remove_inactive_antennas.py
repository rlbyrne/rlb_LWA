import pyuvdata
import os
import numpy as np


# Script to remove "data" from antennas that are not plugged in, i.e. visibilities are zeroed

data_path = "/lustre/rbyrne/LWA_data_20220210"
save_path = f"{data_path}/uvfits_antenna_selected"

filenames = os.listdir(data_path)
use_files = []
use_files = [
    filename
    for filename in filenames
    if len(filename.split(".")) > 1 and filename.split(".")[1] == "uvfits"
]

Nants = 352
autocorr_bls = [(ant_ind, ant_ind) for ant_ind in range(Nants)]
for file_ind, uvfits_file in enumerate(use_files):
    uv = pyuvdata.UVData()
    uv.read_uvfits(
        f"{data_path}/{uvfits_file}", bls=autocorr_bls
    )

    unused_ants = []
    used_ants = []
    for ant_ind in range(Nants):
        ant_name = uv.antenna_names[uv.ant_1_array[ant_ind]]
        avg_autocorr = np.mean(np.abs(uv.data_array[ant_ind, 0, :, :]))
        if avg_autocorr < 1.:
            unused_ants.append(ant_name)
        else:
            used_ants.append(ant_name)

    print(f"Used antennas: {used_ants}")
    print(f"Unused antennas: {unused_ants}")

    uv.write_uvfits(f"{save_path}/{uvfits_file}")