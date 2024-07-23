import pyuvdata
import numpy as np
import os


def divide_file_in_time():
    input_file = "/lustre/gh/2024-03-02/calibration/ruby/46.ms"
    uv = pyuvdata.UVData()
    uv.read_ms(input_file)
    times = list(set(uv.time_array))
    time_ind = 1
    for use_time in times:
        uv_1time = uv.select(times=use_time, inplace=False)
        uv_1time.write_ms(f"/lustre/rbyrne/2024-03-02/46_time{time_ind}.ms")
        time_ind += 1


def run_simulation():

    script_path = "/home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis.py"
    catalog_path = "/fast/rbyrne/skymodels/Gasperin2020_point_sources_plus_48MHz.skyh5"
    beam_file = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits"
    # for time_ind in range(1, 25):
    for time_ind in range(2, 25):
        input_obs = f"/lustre/rbyrne/2024-03-02/46_time{time_ind}.ms"
        output_file = f"/lustre/rbyrne/2024-03-02/calibration_models/46_time{time_ind}_deGasperin_point_sources.ms"
        run_command = f"mpirun -n 20 python {script_path} {catalog_path} {beam_file} {input_obs} {output_file}"
        os.system(run_command)


if __name__ == "__main__":
    run_simulation()
