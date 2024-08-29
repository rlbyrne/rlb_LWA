# Run in CASA with command execfile("partition_ms_files.py")

import os

dirname = "10"
pathname = f"/lustre/rbyrne/2024-03-02/{dirname}"
use_files = ["41", "46", "50", "55", "59", "64", "73", "78", "82"]

time_step = 0
while time_step < 72:
    for filename in use_files:
        if f"{filename}_{dirname}_{str(time_step).zfill(3)}.ms" not in os.listdir(
            pathname
        ):  # Check if file already exists
            partition(
                f"{pathname}/{filename}.ms",
                outputvis=f"{pathname}/{filename}_{dirname}_{str(time_step).zfill(3)}.ms",
                scan=f"{int(time_step * 5 + 1)}~{int(time_step * 5 + 5)}",
            )

    datafile_list = [
        f"{pathname}/{filename}_{dirname}_{str(time_step).zfill(3)}.ms"
        for filename in use_files
    ]
    virtualconcat(
        vis=datafile_list,
        concatvis=f"{pathname}/41-82_{dirname}_{str(time_step).zfill(3)}.ms",
    )
    time_step += 1
