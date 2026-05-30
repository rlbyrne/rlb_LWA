import pyuvdata
import subprocess
import numpy as np
import sys
sys.path.append("/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing")
import LWA_calibrate

use_freqs = np.array(
    [
        "34",
        "44",
        "52",
        "62",
        "72",
        "79",
        "83",
    ]
)
time_slices = [
    "112643-112833",
    "112843-113033",
    "113043-113234",
    "113244-113434",
    "113444-113635",
]
for freq in use_freqs:
    filenames = [f"20260419_{time_slice}_{freq}MHz_17h_cal_peeled.ms" for time_slice in time_slices]
    for filename in filenames:
        subprocess.run(
            ["cp", "-r", f"/lustre/rbyrne/2026-04-19/{filename}", "/fast/rbyrne"],
            check=True
        )
    for file_ind, filename in enumerate(filenames):
        uv_new = pyuvdata.UVData()
        uv_new.read(f"/fast/rbyrne/{filename}")
        uv_new.select(polarizations=[-5, -6])
        if file_ind == 0:
            uv = uv_new
            uv.phase_to_time(np.min(uv.time_array))
        else:
            uv_new.phase_to_time(np.min(uv.time_array))
            uv.fast_concat(uv_new, "blt", inplace=True)
    uv.phase_to_time(np.mean(uv.time_array))
    new_filename = f"20260419_10min_112643-113635_{freq}MHz_17h_cal_peeled"
    uv.write_ms(f"/fast/rbyrne/{new_filename}.ms", clobber=True)
    for filename in filenames:
        subprocess.run(["rm", "-r", f"/fast/rbyrne/{filename}"], check=True)

    LWA_calibrate.image_data(new_filename, f"{new_filename}.ms", niter=50000)
    subprocess.run(["cp", "-r", f"/fast/rbyrne/{new_filename}.ms", "/lustre/rbyrne/2026-04-19"], check=True)
    

