import os
import sys
import numpy as np

target_dir = "/lustre/pipeline/cosmology"
source_dir = "/lustre/pipeline/slow"

copy_freqs = [
    "27",
    "32",
    "36",
    "41",
    "46",
    "50",
    "55",
    "59",
    "64",
    "69",
    "73",
    "78",
    "82",
]

time_ranges = [
    ["013000", "020000"],  # half an hour of calibration data
    ["110000", "140000"],  # three hours of cosmology data
]
use_dates = ["2026-02-07", "2026-02-08"]

dates = np.sort(os.listdir(f"{source_dir}/{copy_freqs[0]}MHz"))
dates = [date for date in dates if date in use_dates]
for date in dates:
    hours = np.sort(os.listdir(f"{source_dir}/{copy_freqs[0]}MHz/{date}"))
    for hour in hours:
        filenames = np.sort(
            os.listdir(f"{source_dir}/{copy_freqs[0]}MHz/{date}/{hour}")
        )
        for filename in filenames:
            timestamp = filename[9:15]
            copy_file = False
            for time_range in time_ranges:
                if int(time_range[0]) < int(timestamp) < int(time_range[1]):
                    copy_file = True
                    break
            if copy_file:
                for freq in copy_freqs:
                    filename = f"{filename[:16]}{freq}{filename[18:]}"
                    if os.path.isdir(
                        f"{source_dir}/{freq}MHz/{date}/{hour}/{filename}"
                    ):
                        # copy_command = f"sudo mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && sudo cp -r {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
                        copy_command = f"mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && cp -r {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
                        print(f"Copying {filename}")
                        os.system(copy_command)
                    else:
                        continue
