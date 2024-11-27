import os
import astropy
from astropy.time import Time
from astropy import units
import pyuvdata
import datetime
import numpy as np

telescope_loc = pyuvdata.telescopes.known_telescope_location("OVRO-LWA")
subband_dirs = np.sort(
    [name for name in os.listdir("/lustre/pipeline/night-time") if name.endswith("MHz")]
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
                    location=pyuvdata.telescopes.known_telescope_location("OVRO-LWA"),
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
