import pyuvdata
import numpy as np
import os
import sys
import datetime


def chunk_in_frequency(uv, filenames, freq_intervals):

    out_filepaths = []
    for interval in np.array(freq_intervals):
        keep_freqs = np.where(
            (uv.freq_array >= interval[0]) & (uv.freq_array <= interval[1])
        )[0]
        uv_freq_band = uv.select(frequencies=uv.freq_array[keep_freqs], inplace=False)

        mean_freq = int(np.mean(uv.freq_array[keep_freqs] / 1e6))
        year = filenames[0][:4]
        month = filenames[0][4:6]
        day = filenames[0][6:8]
        hour = filenames[0][9:11]
        start_minute = filenames[0][11:13]
        start_second = filenames[0][13:15]
        end_minute = filenames[-1][11:13]
        end_second = filenames[-1][13:15]

        # outdir = f"/lustre/pipeline/cosmology/concatenated_data/{mean_freq}MHz/{year}-{month}-{day}/{hour}"
        outdir = (
            f"/fast/rbyrne/concatenated_data/{mean_freq}MHz/{year}-{month}-{day}/{hour}"
        )
        out_filename = f"{year}{month}{day}_{hour}{start_minute}{start_second}-{hour}{end_minute}{end_second}_{mean_freq}MHz.ms"
        if not os.path.isdir(outdir):
            os.system(f"mkdir -p {outdir}")
        uv.write_ms(f"{outdir}/{out_filename}", clobber=True)
        out_filepaths.append(f"{outdir}/{out_filename}")

    return out_filepaths


def delete_original_data(filenames):

    for file_ind, filename in enumerate(filenames):
        for freq_ind, freq in enumerate(use_freqs):
            year = filename[:4]
            month = filename[4:6]
            day = filename[6:8]
            hour = filename[9:11]
            minute = filename[11:13]
            second = filename[13:15]
            path = f"/lustre/pipeline/cosmology/{freq}MHz/{year}-{month}-{day}/{hour}/{year}{month}{day}_{hour}{minute}{second}_{freq}MHz.ms"
            os.system(f"rm -r {path}")


def concatenate(filenames, use_freqs, freq_intervals, delete_orig_data=True):

    # Check if all paths exist
    for file_ind, filename in enumerate(filenames):
        for freq_ind, freq in enumerate(use_freqs):
            year = filename[:4]
            month = filename[4:6]
            day = filename[6:8]
            hour = filename[9:11]
            minute = filename[11:13]
            second = filename[13:15]
            path = f"/lustre/pipeline/cosmology/{freq}MHz/{year}-{month}-{day}/{hour}/{year}{month}{day}_{hour}{minute}{second}_{freq}MHz.ms"
            if not os.path.isdir(path):
                print(f"Error: Path {path} not found.")
                return None

    for file_ind, filename in enumerate(filenames):
        for freq_ind, freq in enumerate(use_freqs):
            year = filename[:4]
            month = filename[4:6]
            day = filename[6:8]
            hour = filename[9:11]
            minute = filename[11:13]
            second = filename[13:15]
            path = f"/lustre/pipeline/cosmology/{freq}MHz/{year}-{month}-{day}/{hour}/{year}{month}{day}_{hour}{minute}{second}_{freq}MHz.ms"

            uv_new = pyuvdata.UVData()
            uv_new.read(path)
            uv_new.select(polarizations=[-5, -6])
            # uv_new.scan_number_array = None  # Added as a workaround for a pyuvdata bug (https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/1595)
            if freq_ind == 0:
                uv_new.phase_to_time(np.min(uv_new.time_array))
                uv_freq_new = uv_new
            else:
                uv_new.phase_to_time(np.min(uv_freq_new.time_array))
                uv_freq_new.fast_concat(uv_new, "freq", inplace=True)
        if file_ind == 0:
            uv = uv_freq_new
        else:
            uv_freq_new.phase_to_time(np.min(uv.time_array))
            uv.fast_concat(uv_freq_new, "blt", inplace=True, run_check=False)

    out_filepaths = chunk_in_frequency(uv, filenames, freq_intervals)
    if delete_orig_data:
        delete_data = True
        for filepath in out_filepaths:
            if not os.path.isdir(filepath):
                delete_data = False
                break
        if delete_data:
            delete_original_data(filenames)


def find_and_concatenate_data(dates, use_freqs, freq_intervals, delete_orig_data=True):

    for date in dates:
        hours = np.sort(
            os.listdir(f"/lustre/pipeline/cosmology/{use_freqs[0]}MHz/{date}")
        )[
            1:
        ]  # Start from hour 12
        for hour in hours:
            filenames = np.sort(
                os.listdir(
                    f"/lustre/pipeline/cosmology/{use_freqs[0]}MHz/{date}/{hour}"
                )
            )
            use_timestamps = []
            use_filenames = []
            for filename in filenames:
                new_time = datetime.datetime(
                    int(filename[:4]),  # year
                    int(filename[4:6]),  # month
                    int(filename[6:8]),  # day
                    int(filename[9:11]),  # hour
                    int(filename[11:13]),  # minute
                    int(filename[13:15]),  # second
                )
                if len(use_timestamps) == 0:
                    use_timestamps.append(new_time)
                    use_filenames.append(filename)
                else:
                    if np.abs(new_time - np.min(use_timestamps)) < datetime.timedelta(
                        minutes=2
                    ):
                        use_timestamps.append(new_time)
                        use_filenames.append(filename)
                if len(use_timestamps) == 12:
                    concatenate(
                        use_filenames,
                        use_freqs,
                        freq_intervals,
                        delete_orig_data=delete_orig_data,
                    )
                    use_timestamps = []
                    use_filenames = []


if __name__ == "__main__":
    use_freqs = [
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
    freq_intervals = [
        [28352050.78125, 41295898.4375],
        [41319824.21875, 48354003.90625],
        [48377929.6875, 56823730.46875],
        [56847656.25, 67398925.78125],
        [67422851.5625, 77615234.375],
        [77639160.15625, 82256835.9375],
        [82280761.71875, 84649414.0625],
    ]
    # dates = np.sort(os.listdir(f"/lustre/pipeline/cosmology/{use_freqs[0]}MHz"))
    dates = ["2026-01-12"]
    # There seems to be a bug where the full frequency range is written out for each file
    find_and_concatenate_data(dates, use_freqs, freq_intervals, delete_orig_data=False)
