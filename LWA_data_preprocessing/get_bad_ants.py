from mnc import anthealth
from astropy.time import Time
import datetime
import sys
import numpy as np

# This exists as a stand-alone function so that it can be run in a separate conda environment
# Run with, for example, subprocess.getoutput("conda run -n deployment python get_bad_ants.py 2025 1 17")


def get_bad_ants(year, month, day, time_interval_tolerance_days=1):
    file_time = Time(
        datetime.datetime(
            year,
            month,
            day,
            0,  # hour
            0,  # minute
            0,  # second
        )
    )
    badants = anthealth.get_badants(
        "selfcorr", time=Time(file_time.strftime("%Y-%m-%d %H:%M:%S"), format="iso").mjd
    )
    if (
        np.abs(Time(badants[0], format="mjd") - file_time)
        > time_interval_tolerance_days
    ):  # bandant data does not exist for the date used
        print(f"get_bandants output: (np.nan, [])")
    else:
        print(f"get_bandants output: {badants}")


if __name__ == "__main__":
    args = sys.argv
    obs_year = sys.argv[1]
    obs_month = sys.argv[2]
    obs_day = sys.argv[3]
    get_bad_ants(int(obs_year), int(obs_month), int(obs_day))
