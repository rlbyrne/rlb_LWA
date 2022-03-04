import pyuvdata
import os
import numpy as np
import matplotlib.pyplot as plt


data_path = "/lustre/rbyrne/LWA_data_20220210"
save_path = f"{data_path}/autocorrelation_plots"

filenames = os.listdir(data_path)
use_files = []
use_files = [
    filename
    for filename in filenames
    if len(filename.split(".")) > 1 and filename.split(".")[1] == "uvfits"
]

Nants = 352
autocorr_bls = [(ant_ind, ant_ind) for ant_ind in range(Nants)]
for uvfits_file in use_files:
    uv = pyuvdata.UVData()
    uv.read_uvfits(
        f"{data_path}/{uvfits_file}", bls=autocorr_bls, polarizations=[-5, -6]
    )
    times = list(set(uv.time_array))
    for time_ind, time_val in enumerate(times):
        uv_time_selected = uv.select(times=time_val, inplace=False)
        for pol_ind, pol_name in enumerate(["xx", "yy"]):
            for ant_ind in range(Nants):
                ant_name = uv_time_selected.antenna_names[uv_time_selected.ant_1_array[ant_ind]]
                plt.plot(
                    uv_time_selected.freq_array[0, :],
                    np.real(uv_time_selected.data_array[ant_ind, 0, :, pol_ind]),
                    '-o', markersize=2, linewidth=1, label=ant_name
                )
            plt.legend(prop={'size': 4})
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Autocorr. Power")
            plt.savefig(
                f"{save_path}/LWA_data_20220210_autocorr_{pol_name}_{time_ind:05d}"
            )
            plt.close()
