import os
import sys

target_dir = "/lustre/pipeline/cosmology"
source_dir = "/lustre/pipeline/slow"

copy_freqs = [
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
    ["120000", "150000"],  # three hours of cosmology data
]

for freq in copy_freqs:
    dates = os.listdir(f"{source_dir}/{freq}MHz")
    for date in dates:
        hours = os.listdir(f"{source_dir}/{freq}MHz/{date}")
        for hour in hours:
            filenames = os.listdir(f"{source_dir}/{freq}MHz/{date}/{hour}")
            for filename in filenames:
                timestamp = filename[9:15]
                copy_file = False
                for time_range in time_ranges:
                    if int(time_range[0]) < int(timestamp) < int(time_range[1]):
                        copy_file = True
                        break
                if copy_file:
                    copy_command = f"sudo mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && sudo cp -r {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
                    os.system(copy_command)
