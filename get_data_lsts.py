import os
import astropy
from astropy.time import Time
from astropy import units
import pyuvdata
import datetime
import numpy as np
import pandas as pd


def get_observation_lsts():

    telescope_loc = pyuvdata.telescopes.known_telescope_location("OVRO-LWA")
    subband_dirs = np.sort(
        [
            name
            for name in os.listdir("/lustre/pipeline/night-time")
            if name.endswith("MHz")
        ]
    )
    use_subband_dirs = [
        name for name in subband_dirs if int(name[:2]) >= 41
    ]  # Only use highband data
    lst_start = astropy.coordinates.Longitude(11, unit="hour")
    lst_end = astropy.coordinates.Longitude(14, unit="hour")

    output_filename_all = f"/lustre/rbyrne/datafile_lsts.csv"
    output_filename_lst_cut = f"/lustre/rbyrne/datafile_lsts_11-14.csv"
    with open(output_filename_all, "w") as f:
        f.write("filename, full path, LST, in LST range? (T/F), nighttime? (T/F) \n")
    with open(output_filename_lst_cut, "w") as f:
        f.write("filename, full path, LST, in LST range? (T/F), nighttime? (T/F) \n")
    filenames = []
    full_filepaths = []
    lsts = []
    in_lst_range = []
    in_nighttime = []
    for subband in use_subband_dirs:
        day_dirs = np.sort(os.listdir(f"/lustre/pipeline/night-time/{subband}"))
        for day in day_dirs:
            hours = np.sort(os.listdir(f"/lustre/pipeline/night-time/{subband}/{day}"))
            for hour in hours:
                files = np.sort(
                    [
                        filename
                        for filename in os.listdir(
                            f"/lustre/pipeline/night-time/{subband}/{day}/{hour}"
                        )
                        if filename.endswith(".ms")
                    ]
                )
                for filename in files:
                    file_time = datetime.datetime(
                        int(filename[:4]),  # year
                        int(filename[4:6]),  # month
                        int(filename[6:8]),  # day
                        int(filename[9:11]),  # hour
                        int(filename[11:13]),  # minute
                        int(filename[13:15]),  # second
                    )
                    observing_time = Time(
                        file_time,
                        scale="utc",
                        location=pyuvdata.telescopes.known_telescope_location(
                            "OVRO-LWA"
                        ),
                    )
                    lst = observing_time.sidereal_time("apparent")
                    filenames.append(filename)
                    full_filepaths.append(
                        f"/lustre/pipeline/night-time/{subband}/{day}/{hour}/{filename}"
                    )
                    lsts.append(lst)
                    in_lst_range.append(lst > lst_start and lst < lst_end)
                    sun_loc = astropy.coordinates.get_sun(observing_time)
                    sun_altaz = sun_loc.transform_to(
                        astropy.coordinates.AltAz(
                            obstime=observing_time, location=telescope_loc
                        )
                    )
                    in_nighttime.append(
                        sun_altaz.alt < -10 * units.deg
                    )  # Sun must be lower than 10 degrees below the horizon
                    with open(output_filename_all, "a") as f:
                        f.write(
                            f"{filename}, /lustre/pipeline/night-time/{subband}/{day}/{hour}/{filename}, {lst}, {lst > lst_start and lst < lst_end}, {sun_altaz.alt < -10 * units.deg} \n"
                        )
                    if (
                        lst > lst_start
                        and lst < lst_end
                        and sun_altaz.alt < -10 * units.deg
                    ):
                        with open(output_filename_lst_cut, "a") as f:
                            f.write(
                                f"{filename}, /lustre/pipeline/night-time/{subband}/{day}/{hour}/{filename}, {lst}, {lst > lst_start and lst < lst_end}, {sun_altaz.alt < -10 * units.deg} \n"
                            )


def find_missing_subbands():

    df = pd.read_csv("/lustre/rbyrne/datafile_lsts_11-14.csv")
    use_subbands = [
        "41MHz",
        "46MHz",
        "50MHz",
        "55MHz",
        "59MHz",
        "64MHz",
        "69MHz",
        "73MHz",
        "78MHz",
        "82MHz",
    ]

    filenames = []
    file_times = []
    for ind in range(len(df["filename"])):
        filename = df["filename"][ind]
        filenames.append(filename)
        file_times.append(
            datetime.datetime(
                int(filename[:4]),  # year
                int(filename[4:6]),  # month
                int(filename[6:8]),  # day
                int(filename[9:11]),  # hour
                int(filename[11:13]),  # minute
                int(filename[13:15]),  # second
            )
        )
    keep_files = np.full(len(df["filename"]), True, dtype=bool)
    for file_ind, filename in enumerate(filenames):
        if (file_ind + 1) % 100 == 0:
            print(f"Processing file {file_ind+1} of {len(filenames)}.")
        if filename.endswith(f"{use_subbands[0]}.ms"):
            same_time_file_inds = np.where(
                np.abs(np.array(file_times) - file_times[file_ind])
                < datetime.timedelta(seconds=5)
            )[0]
            if len(same_time_file_inds) < len(use_subbands):
                keep_files[same_time_file_inds] = False
            else:
                same_time_filenames = np.array(filenames)[same_time_file_inds]
                all_files_present = True
                for subband in use_subbands:
                    if (
                        np.sum(
                            [
                                use_file.endswith(f"{subband}.ms")
                                for use_file in same_time_filenames
                            ]
                        )
                        == 0
                    ):
                        all_files_present = False
                        break
                if not all_files_present:
                    keep_files[same_time_file_inds] = False
    if np.min(keep_files):
        print("All subbands are present for all timesteps.")
    else:
        df["all subbands exist? (T/F)"] = keep_files
        df.to_csv("/lustre/rbyrne/datafile_lsts_11-14_full_band_col.csv", index=False)
        df.iloc[np.where(keep_files)[0]].to_csv(
            "/lustre/rbyrne/datafile_lsts_11-14_full_band_selected.csv", index=False
        )


def add_2025_data():

    telescope_loc = pyuvdata.telescopes.known_telescope_location("OVRO-LWA")
    subband_dirs = np.sort(
        [
            name
            for name in os.listdir("/lustre/pipeline/cosmology")
            if name.endswith("MHz")
        ]
    )
    use_subband_dirs = [
        name for name in subband_dirs if int(name[:2]) >= 41
    ]  # Only use highband data
    lst_start = astropy.coordinates.Longitude(11, unit="hour")
    lst_end = astropy.coordinates.Longitude(14, unit="hour")

    output_filename_all = f"/lustre/rbyrne/datafile_lsts.csv"
    output_filename_lst_cut = f"/lustre/rbyrne/datafile_lsts_11-14.csv"
    output_filename_full_band_selected = f"/lustre/rbyrne/datafile_lsts_11-14_full_band_selected.csv"

    filenames = []
    full_filepaths = []
    lsts = []
    in_lst_range = []
    in_nighttime = []
    for subband in use_subband_dirs:
        day_dirs = np.sort(os.listdir(f"/lustre/pipeline/cosmology/{subband}"))
        for day in day_dirs:
            hours = np.sort(os.listdir(f"/lustre/pipeline/cosmology/{subband}/{day}"))
            for hour in hours:
                files = np.sort(
                    [
                        filename
                        for filename in os.listdir(
                            f"/lustre/pipeline/cosmology/{subband}/{day}/{hour}"
                        )
                        if filename.endswith(".ms")
                    ]
                )
                for filename in files:
                    file_time = datetime.datetime(
                        int(filename[:4]),  # year
                        int(filename[4:6]),  # month
                        int(filename[6:8]),  # day
                        int(filename[9:11]),  # hour
                        int(filename[11:13]),  # minute
                        int(filename[13:15]),  # second
                    )
                    observing_time = Time(
                        file_time,
                        scale="utc",
                        location=pyuvdata.telescopes.known_telescope_location(
                            "OVRO-LWA"
                        ),
                    )
                    lst = observing_time.sidereal_time("apparent")
                    filenames.append(filename)
                    full_filepaths.append(
                        f"/lustre/pipeline/cosmology/{subband}/{day}/{hour}/{filename}"
                    )
                    lsts.append(lst)
                    in_lst_range.append(lst > lst_start and lst < lst_end)
                    sun_loc = astropy.coordinates.get_sun(observing_time)
                    sun_altaz = sun_loc.transform_to(
                        astropy.coordinates.AltAz(
                            obstime=observing_time, location=telescope_loc
                        )
                    )
                    in_nighttime.append(
                        sun_altaz.alt < -10 * units.deg
                    )  # Sun must be lower than 10 degrees below the horizon
                    with open(output_filename_all, "a") as f:
                        f.write(
                            f"{filename}, /lustre/pipeline/cosmology/{subband}/{day}/{hour}/{filename}, {lst}, {lst > lst_start and lst < lst_end}, {sun_altaz.alt < -10 * units.deg} \n"
                        )
                    if (
                        lst > lst_start
                        and lst < lst_end
                        and sun_altaz.alt < -10 * units.deg
                    ):
                        with open(output_filename_lst_cut, "a") as f:
                            f.write(
                                f"{filename}, /lustre/pipeline/cosmology/{subband}/{day}/{hour}/{filename}, {lst}, {lst > lst_start and lst < lst_end}, {sun_altaz.alt < -10 * units.deg} \n"
                            )


if __name__ == "__main__":
    add_2025_data()
