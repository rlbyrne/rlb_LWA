import pathlib
import numpy as np
import datetime
import pyuvdata
import os


def populate_data_dict(data_paths):

    filenames = [path.split("/")[-1] for path in data_paths]
    times = np.array(
        [
            datetime.datetime.strptime(filename[:15], "%Y%m%d_%H%M%S")
            for filename in filenames
        ],
        dtype="datetime64[s]",
    )
    freqs = np.array([filename[16:18] for filename in filenames])
    data_dict = {
        "paths": data_paths,
        "freqs": freqs,
        "times": times,
    }
    return data_dict


def get_time_inds(data_dict, time_interval_minutes=2, time_integration_s=10):
    times = data_dict["times"]
    interval = np.timedelta64(int(time_interval_minutes), "m")
    half_interval = interval / 2

    time_chunk_centers = np.arange(
        np.min(times) + (interval - np.timedelta64(int(time_integration_s), "s")) / 2,
        np.max(times) + half_interval,
        interval,
    )

    time_inds = np.abs(times[:, None] - time_chunk_centers[None, :]).argmin(axis=1)
    data_dict["time_inds"] = time_inds
    return data_dict


def get_orig_freq_intervals():
    freq_interval_dict = {}
    freq_interval_dict["27"] = np.array(
        [np.float64(27179687.5), np.float64(31749511.71875)]
    )
    freq_interval_dict["32"] = np.array(
        [np.float64(31773437.5), np.float64(36343261.71875)]
    )
    freq_interval_dict["36"] = np.array(
        [np.float64(36367187.5), np.float64(40937011.71875)]
    )
    freq_interval_dict["41"] = np.array(
        [np.float64(40960937.5), np.float64(45530761.71875)]
    )
    freq_interval_dict["46"] = np.array(
        [np.float64(45554687.5), np.float64(50124511.71875)]
    )
    freq_interval_dict["50"] = np.array(
        [np.float64(50148437.5), np.float64(54718261.71875)]
    )
    freq_interval_dict["55"] = np.array(
        [np.float64(54742187.5), np.float64(59312011.71875)]
    )
    freq_interval_dict["59"] = np.array(
        [np.float64(59335937.5), np.float64(63905761.71875)]
    )
    freq_interval_dict["64"] = np.array(
        [np.float64(63929687.5), np.float64(68499511.71875)]
    )
    freq_interval_dict["69"] = np.array(
        [np.float64(68523437.5), np.float64(73093261.71875)]
    )
    freq_interval_dict["73"] = np.array(
        [np.float64(73117187.5), np.float64(77687011.71875)]
    )
    freq_interval_dict["78"] = np.array(
        [np.float64(77710937.5), np.float64(82280761.71875)]
    )
    freq_interval_dict["82"] = np.array(
        [np.float64(82304687.5), np.float64(86874511.71875)]
    )
    return freq_interval_dict


def get_new_freq_intervals():
    # Determined by where the equalization coefficients are held constant
    freq_interval_dict = {}
    freq_interval_dict["34"] = np.array([28352050.78125, 41295898.4375])
    freq_interval_dict["44"] = np.array([41319824.21875, 48354003.90625])
    freq_interval_dict["52"] = np.array([48377929.6875, 56823730.46875])
    freq_interval_dict["62"] = np.array([56847656.25, 67398925.78125])
    freq_interval_dict["72"] = np.array([67422851.5625, 77615234.375])
    freq_interval_dict["79"] = np.array([77639160.15625, 82256835.9375])
    freq_interval_dict["83"] = np.array([82280761.71875, 84649414.0625])
    return freq_interval_dict


def get_contributing_freqs(new_freq_interval_dict, orig_freq_interval_dict):
    contributing_freqs = []
    for new_freq in new_freq_interval_dict.keys():
        new_interval = new_freq_interval_dict[new_freq]
        contributing_freqs.append(
            [
                freq
                for freq, orig_interval in orig_freq_interval_dict.items()
                if orig_interval[0] <= new_interval[1]
                and orig_interval[1] >= new_interval[0]
            ]
        )
    return contributing_freqs


def copy_files_to_tmp(filepaths, tmp_dir):

    if not os.path.isdir(tmp_dir):
        os.system(f"sudo mkdir -p {tmp_dir}; sudo chmod a+w {tmp_dir}")

    tmp_filepaths = []
    for filename in filepaths:
        os.system(f"cp -r {filename} {tmp_dir}")
        new_filename = f"{tmp_dir}/{filename.split('/')[-1]}"
        if os.path.isdir(new_filename):
            tmp_filepaths.append(new_filename)
        else:
            print(f"ERROR: Moving file {filename} to {tmp_dir} failed.")
            tmp_filepaths.append(filename)

    return tmp_filepaths


def get_output_filename(data_dict, use_inds, output_freq, output_dir, is_tmp_dir=False):

    times = np.array([data_dict["times"][ind] for ind in use_inds])

    if is_tmp_dir:
        out_filename = f"{output_dir}/{np.min(times).item().strftime('%Y%m%d_%H%M%S')}-{np.max(times).item().strftime('%H%M%S')}_{output_freq}MHz.ms"
    else:
        out_filename = f"{output_dir}/{output_freq}MHz/{np.min(times).item().strftime('%Y-%m-%d')}/{np.min(times).item().strftime('%H')}/{np.min(times).item().strftime('%Y%m%d_%H%M%S')}-{np.max(times).item().strftime('%H%M%S')}_{output_freq}MHz.ms"

    return out_filename


def concatenate_files(
    data_dict,
    use_inds,
    output_freq,
    new_freq_interval_dict,
    output_dir,
    tmp_dir=None,
    refresh=False,
):

    output_filename = get_output_filename(data_dict, use_inds, output_freq, output_dir)

    if not refresh:
        if os.path.isdir(output_filename):
            print(f"File {output_filename} already exists. Skipping.")
            return 3

    filepaths = np.array([data_dict["paths"][ind] for ind in use_inds])
    if tmp_dir is None:
        use_filepaths = filepaths
    else:
        use_filepaths = copy_files_to_tmp(filepaths, tmp_dir)

    frequencies = np.array([data_dict["freqs"][ind] for ind in use_inds])
    times = np.array([data_dict["times"][ind] for ind in use_inds])

    for time_ind, time in enumerate(np.unique(times)):  # Iterate over times
        for freq_ind, freq in enumerate(
            np.unique(frequencies)
        ):  # Iterate over frequencies
            path_use = use_filepaths[
                np.where((times == time) & (frequencies == freq))[0][0]
            ]

            uv_new = pyuvdata.UVData()
            try:
                print(f"Reading file {path_use}")
                uv_new.read(path_use)
            except:
                print(f"WARNING: Error reading file {path_use}. Skipping.")
                return 2

            # uv_new.select(polarizations=[-5, -6])
            uv_new.scan_number_array = None  # Added as a workaround for a pyuvdata bug (https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/1595)
            if freq_ind == 0:
                uv_new.phase_to_time(np.min(uv_new.time_array))
                uv_single_freq = uv_new
            else:
                uv_new.phase_to_time(np.min(uv_single_freq.time_array))
                uv_single_freq.fast_concat(uv_new, "freq", inplace=True)
        if time_ind == 0:
            uv = uv_single_freq
        else:
            uv_single_freq.phase_to_time(np.min(uv.time_array))
            uv.fast_concat(uv_single_freq, "blt", inplace=True, run_check=False)

    keep_freqs = np.where(
        (uv.freq_array >= new_freq_interval_dict[output_freq][0])
        & (uv.freq_array <= new_freq_interval_dict[output_freq][1])
    )[0]
    uv.select(frequencies=uv.freq_array[keep_freqs], inplace=True)

    if tmp_dir is None:
        use_output_dir = output_dir
        use_output_filename = output_filename
    else:
        use_output_dir = tmp_dir
        use_output_filename = get_output_filename(
            data_dict, use_inds, output_freq, use_output_dir, is_tmp_dir=True
        )

    # Create output directory if needed
    outdir = "/".join(use_output_filename.split("/")[:-1])
    if not os.path.isdir(outdir):
        os.system(f"sudo mkdir -p {outdir}; sudo chmod a+w {outdir}")

    uv.write_ms(use_output_filename, clobber=True)
    if not os.path.isdir(use_output_filename):
        print(f"Error writing file {use_output_filename}.")
        return 4

    if tmp_dir is not None:
        # Create output directory if needed
        outdir = "/".join(output_filename.split("/")[:-1])
        if not os.path.isdir(outdir):
            os.system(f"sudo mkdir -p {outdir}; sudo chmod a+w {outdir}")

        # Move concatenated data
        os.system(f"mv {use_output_filename} {outdir}")
        if not os.path.isdir(output_filename):
            print(f"Error moving file to {output_filename}.")
            return 4

        # Delete copied data
        for filename in use_filepaths:
            if filename not in filepaths:  # Confirm that the file is a copy
                os.system(f"rm -r {filename}")  # Delete temporary files

    print(f"New file written to {output_filename}")

    return 0


def run_concatenate_data(
    date,
    orig_dir="/lustre/pipeline/cosmology",
    output_dir="/lustre/pipeline/cosmology/concatenated_data",
    delete_files=True,
):

    new_freq_interval_dict = get_new_freq_intervals()
    orig_freq_interval_dict = get_orig_freq_intervals()
    contributing_freqs = get_contributing_freqs(
        new_freq_interval_dict, orig_freq_interval_dict
    )

    data_paths = []
    for freq in orig_freq_interval_dict.keys():
        data_paths.extend(
            [
                str(p)
                for p in pathlib.Path(f"{orig_dir}/{freq}MHz/{date}").rglob("*.ms")
                if p.is_dir()
            ]
        )

    data_dict = populate_data_dict(data_paths)
    data_dict = get_time_inds(data_dict)

    # Status strings:
    # 0: Success
    # 1: Not all files exist
    # 2: Not all files readable
    # 3: Concatenated file already exists
    # 4: Unknown error
    delete_file_conditions = [
        0,
        1,
        3,
    ]  # Only delete files if all meet these conditions. Used only if delete_files=True

    for time_ind in list(set(data_dict["time_inds"])):
        file_inds_time_match = np.where(data_dict["time_inds"] == time_ind)[0]
        status = []
        for freq_ind, output_freq in enumerate(new_freq_interval_dict.keys()):
            file_inds_freq_match = np.where(
                np.array(
                    [
                        freq in contributing_freqs[freq_ind]
                        for freq in data_dict["freqs"]
                    ]
                )
            )[0]
            use_inds = np.intersect1d(file_inds_time_match, file_inds_freq_match)
            if len(use_inds) != len(contributing_freqs[freq_ind]) * 12:
                print(
                    f"ERROR: Not all files are present for time {time_ind} and frequency {output_freq}MHz."
                )
                status.append(1)
            else:
                new_status = concatenate_files(
                    data_dict,
                    use_inds,
                    output_freq,
                    new_freq_interval_dict,
                    output_dir,
                    tmp_dir="/fast/rbyrne/data_concatenation_tmp",
                )
                status.append(new_status)
        if delete_files:
            if np.min([use_status in delete_file_conditions for use_status in status]):
                for file_ind in file_inds_time_match:
                    filename = data_dict["paths"][file_ind]
                    print(f"Deleting {filename}")
                    os.system(f"sudo rm -r {filename}")


if __name__ == "__main__":
    run_concatenate_data("2026-04-19", delete_files=False)
