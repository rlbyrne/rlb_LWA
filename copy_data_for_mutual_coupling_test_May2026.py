import os
import sys
import numpy as np
import schedule
import time
import astropy.units as u
import astropy.time
from astropy.coordinates import EarthLocation
import pyuvdata


def copy_data_no_tar(
    copy_paths,
    target_dir="/lustre/pipeline/cosmology",
    source_dir="/lustre/pipeline/slow",
):
    for path in copy_paths:
        if os.path.isdir(path) or os.path.isfile(path):
            path_split = path.split("/")
            source_dir = "/".join(path_split[:-4])
            intermediate_dirs = "/".join(path_split[-4:-1])
            filename = path_split[-1]
            copy_command = f"sudo mkdir -p {target_dir}/{intermediate_dirs} && sudo mv {source_dir}/{intermediate_dirs}/{filename} {target_dir}/{intermediate_dirs}"
            print(f"Moving {filename}")
            os.system(copy_command)
        else:
            continue


def copy_data(
    target_dir="/lustre/rbyrne/dsa_dish_pointing_test_May2026",
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
        [204000, 213810],
    ],
    use_dates=["2026-05-05"],
):

    copy_paths = []
    for freq in copy_freqs:
        dates = np.sort(os.listdir(f"{source_dir}/{freq}MHz"))
        if use_dates is not None:
            dates = [date for date in dates if date in use_dates]
        for date in dates:
            hours = np.sort(os.listdir(f"{source_dir}/{freq}MHz/{date}"))
            for hour in hours:
                filenames = np.sort(os.listdir(f"{source_dir}/{freq}MHz/{date}/{hour}"))
                for filename in filenames:
                    timestamp = filename[9:15]
                    copy_file = False
                    for time_range in time_ranges:
                        if int(time_range[0]) < int(timestamp) < int(time_range[1]):
                            copy_file = True
                            break
                    if copy_file:
                        filename = f"{filename[:16]}{freq}{filename[18:]}"
                        copy_paths.append(
                            f"{source_dir}/{freq}MHz/{date}/{hour}/{filename}"
                        )

    copy_data_no_tar(
        copy_paths,
        target_dir=target_dir,
        source_dir=source_dir,
    )


def extract_autocorrs():

    use_filenames = []
    data_dir = "/lustre/rbyrne/dsa_dish_pointing_test_May2026"
    date = "2026-05-05"
    freq_dirs = os.listdir(data_dir)
    for freq in freq_dirs:
        hours = os.listdir(f"{data_dir}/{freq}/{date}")
        for hour in hours:
            files = [
                filename
                for filename in os.listdir(f"{data_dir}/{freq}/{date}/{hour}")
                if filename.endswith(".ms.tar")
            ]
            use_filenames.extend(
                [f"{data_dir}/{freq}/{date}/{hour}/{filename}" for filename in files]
            )

    for filename in use_filenames:
        file_dir = "/".join(filename.split("/")[:-1])
        # untar
        os.system(f"tar -xvf {filename} -C {file_dir}")

        untar_filename = filename.strip(".tar")
        uv = pyuvdata.UVData()
        uv.read(untar_filename)

        uv.select(antenna_names=["LWA364", "LWA363", "LWA362"])
        uv.select(ant_str="auto")

        out_filename = f"{untar_filename.strip('.ms')}.uvh5"
        uv.write_uvh5(out_filename, clobber=True)

        # Delete original data
        if os.path.isfile(out_filename):
            if os.path.getsize(out_filename) > 90000:
                os.system(f"rm -r {untar_filename}")
                os.system(f"rm {filename}")


if __name__ == "__main__":
    extract_autocorrs()
