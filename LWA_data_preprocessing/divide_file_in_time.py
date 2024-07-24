import pyuvdata
import numpy as np
import os


use_filenames = [
    "18",
    "23",
    "27",
    "36",
    "41",
    "46",
    "50",
    "55",
    "59",
    "64",
    "73",
    "78",
    "82",
]

# Break files into time chunks
for file_name in use_filenames:
    input_file = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
    uv = pyuvdata.UVData()
    uv.read_ms(input_file)
    times = list(set(uv.time_array))
    time_ind = 1
    for use_time in times:
        uv_1time = uv.select(times=use_time, inplace=False)
        uv_1time.write_ms(f"/lustre/rbyrne/2024-03-02/{file_name}_time{time_ind}.ms")
        time_ind += 1

# Run simulations
script_path = "/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py"
beam_file = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
for file_name in use_filenames:
    catalog_path = (
        f"/fast/rbyrne/skymodels/Gasperin2020_point_sources_plus_{file_name}.skyh5"
    )
    for time_ind in range(1, 25):
        input_obs = f"/lustre/rbyrne/2024-03-02/{file_name}_time{time_ind}.ms"
        output_file = f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_time{time_ind}_deGasperin_point_sources.ms"
        run_command = f"mpirun -n 20 python {script_path} {catalog_path} {beam_file} {input_obs} {output_file}"
        os.system(run_command)

# Combine results
for file_name in use_filenames:
    for time_ind in range(1, 25):
        uv_new = pyuvdata.UVData()
        uv_new.read_ms(f"/lustre/rbyrne/2024-03-02/{file_name}_time{time_ind}.ms")
        if time_ind == 0:
            uv = uv_new
        else:
            uv.fast_concat(uv_new, inplace=True)
    uv.phase_to_time(np.mean(uv.time_array))
    uv.write_ms(
        f"/lustre/rbyrne/2024-03-02/calibration_models/{file_name}_deGasperin_point_sources.ms"
    )
