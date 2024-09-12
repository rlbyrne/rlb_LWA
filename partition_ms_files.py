# Run in CASA with command execfile("partition_ms_files.py")

import os

dirname = "10"
pathname = f"/lustre/rbyrne/2024-03-02/{dirname}"
use_files = ["41", "46", "50", "55", "59", "64", "73", "78", "82"]

time_step = 0
use_n_timesteps = 30
while time_step < 72:
    output_filename = f"41-82_{dirname}_{str(time_step+1).zfill(3)}_5min.ms"
    if output_filename in os.listdir(pathname):
        print(f"{output_filename} exists. Skipping.")
        time_step += 1
    else:
        for filename in use_files:
            if f"{filename}_{dirname}_{str(time_step).zfill(3)}.ms" not in os.listdir(
                pathname
            ):  # Check if file already exists
                print(f"Partitioning file {filename}.ms")
                partition(
                    f"{pathname}/{filename}.ms",
                    outputvis=f"{pathname}/{filename}_{dirname}_{str(time_step+1).zfill(3)}.ms",
                    scan=f"{int(time_step * use_n_timesteps + 1)}~{int(time_step * use_n_timesteps + use_n_timesteps)}",
                )

        datafile_list = [
            f"{pathname}/{filename}_{dirname}_{str(time_step+1).zfill(3)}.ms"
            for filename in use_files
        ]
        print(f"Concatenating files and saving to {output_filename}")
        virtualconcat(
            vis=datafile_list,
            concatvis=output_filename,
        )
        time_step += 1
