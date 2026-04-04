import os
import sys
import numpy as np
import schedule
import time


def copy_data(
    target_dir="/lustre/pipeline/cosmology",
    source_dir="/lustre/pipeline/slow",
    copy_freqs=[
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
    ],
    time_ranges=[
        ["123000", "130000"],  # half an hour of calibration data
        ["070000", "100000"],  # three hours of cosmology data
    ],
    use_dates=None,
):

    dates = np.sort(os.listdir(f"{source_dir}/{copy_freqs[0]}MHz"))
    if use_dates is not None:
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
                            copy_command = f"mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && mv {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
                            print(f"Copying {filename}")
                            os.system(copy_command)
                        else:
                            continue


schedule.every().hour.do(copy_data)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute