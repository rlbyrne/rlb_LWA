import os
import sys
import numpy as np

def copy_data(
    target_dir="/lustre/rbyrne/requant_paper_spectra",
    source_dir="/lustre/pipeline/slow",
    copy_freqs=[
        "13",
        "18",
        "23",
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
        ["062500", "062600"],
    ],
    use_dates=["2026-04-15"],
):

    for freq in copy_freqs:
        dates = np.sort(os.listdir(f"{source_dir}/{freq}MHz"))
        if use_dates is not None:
            dates = [date for date in dates if date in use_dates]
        for date in dates:

            hours = np.sort(os.listdir(f"{source_dir}/{freq}MHz/{date}"))
            for hour in hours:
                filenames = np.sort(
                    os.listdir(f"{source_dir}/{freq}MHz/{date}/{hour}")
                )
                for filename in filenames:
                    timestamp = filename[9:15]
                    copy_file = False
                    for time_range in time_ranges:
                        if int(time_range[0]) < int(timestamp) < int(time_range[1]):
                            copy_file = True
                            break
                    if copy_file:
                        filename = f"{filename[:16]}{freq}{filename[18:]}"
                        if os.path.isdir(
                            f"{source_dir}/{freq}MHz/{date}/{hour}/{filename}"
                        ):
                            # copy_command = f"sudo mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && sudo cp -r {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
                            copy_command = f"sudo mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && sudo cp -r {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
                            print(f"Moving {filename}")
                            os.system(copy_command)
                        else:
                            continue

if __name__=="__main__":
    copy_data()