import os
import sys
import numpy as np
import schedule
import time
import astropy.units as u
import astropy.time
from astropy.coordinates import EarthLocation

# Needs to be run as pipeline user

def get_utc_from_lst(
    target_lst_hrs,
    date_str,  # Format "YYYY-MM-DD"
    longitude_deg=-118.3
):

    loc = EarthLocation.from_geodetic(lon=longitude_deg * u.deg, lat=0 * u.deg)
    reference_time = astropy.time.Time(f"{date_str[:4]}-{date_str[5:7]}-{date_str[8:10]} 12:00:00", scale='utc', location=loc)
    reference_lst = reference_time.sidereal_time('mean').hour
    lst_diff = (target_lst_hrs - reference_lst + 12) % 24 - 12
    utc = reference_time + (lst_diff * 0.99726958) * u.hour
    if isinstance(utc.datetime, (list, np.ndarray)):
        array_shape = np.shape(utc)
        utc_str = np.array([f"{use_utc.datetime.hour:02d}{use_utc.datetime.minute:02d}{use_utc.datetime.second:02d}" for use_utc in utc.flatten()])
        utc_str = np.reshape(utc_str, shape=array_shape)
    else:
        utc_str = f"{utc.datetime.hour:02d}{utc.datetime.minute:02d}{utc.datetime.second:02d}"
    
    return utc_str


def copy_data(
    target_dir="/lustre/pipeline/cosmology",
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
    lst_ranges=[
        [11.88, 14.88],  # 3 hours of cosmology data
        [17.38, 17.88],  # half an hour of calibration data
    ],
    use_dates=None,
):

    copy_paths = []
    for freq in copy_freqs:
        dates = np.sort(os.listdir(f"{source_dir}/{freq}MHz"))
        if use_dates is not None:
            dates = [date for date in dates if date in use_dates]
        for date in dates:

            # Convert from LST to UTC
            time_ranges = get_utc_from_lst(lst_ranges, date, longitude_deg=-118.3)

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
                        copy_paths.append(f"{source_dir}/{freq}MHz/{date}/{hour}/{filename}")

    time.sleep(120) # wait 2 minutes to prevent moving files mid-write
    for path in copy_paths:
        if os.path.isdir(path):
            path_split = path.split("/")
            source_dir = "/".join(path_split[:-4])
            intermediate_dirs = "/".join(path_split[-4:-1])
            filename = path_split[-1]
            # copy_command = f"sudo mkdir -p {target_dir}/{freq}MHz/{date}/{hour} && sudo cp -r {source_dir}/{freq}MHz/{date}/{hour}/{filename} {target_dir}/{freq}MHz/{date}/{hour}"
            copy_command = f"tar -czf {source_dir}/{intermediate_dirs}/{filename}.tar.gz {source_dir}/{intermediate_dirs}/{filename} && mkdir -p {target_dir}/{intermediate_dirs} && mv {source_dir}/{intermediate_dirs}/{filename}.tar.gz {target_dir}/{intermediate_dirs} && tar -xzvf {target_dir}/{intermediate_dirs}/{filename}.tar.gz"
            print(f"Moving {filename}")
            os.system(copy_command)
        else:
            continue


schedule.every().hour.do(copy_data)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute