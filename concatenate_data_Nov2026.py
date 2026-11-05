import argparse
import datetime
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pyuvdata

# Status codes returned for each (time chunk, output band)
STATUS_SUCCESS = 0
STATUS_MISSING = 1  # Not all input files present
STATUS_UNREADABLE = 2  # Not all input files readable
STATUS_EXISTS = 3  # Concatenated file already exists
STATUS_ERROR = 4  # Unknown error

# Input files may only be deleted if every band for the chunk is in this set.
# STATUS_MISSING is deliberately excluded: an incomplete chunk (late data, a
# chunk spanning midnight, a missing file) must never be deleted.
DELETE_OK_STATUSES = (STATUS_SUCCESS, STATUS_EXISTS)

FILES_PER_FREQ_PER_CHUNK = 12
FILENAME_RE = re.compile(r"^(\d{8}_\d{6})_(\d{2})MHz\.ms(?:\.tar)?$")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_dir(path):
    """Create a directory (and parents), using sudo only if needed.

    Directories created via sudo are chowned back to the current user so that
    later cleanup does not require root. The leaf is made world-writable.
    """
    if os.path.isdir(path):
        return

    missing = []
    p = os.path.abspath(path)
    while not os.path.isdir(p):
        missing.append(p)
        p = os.path.dirname(p)

    try:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, os.stat(path).st_mode | 0o002)
    except PermissionError:
        subprocess.run(["sudo", "mkdir", "-p", path], check=True)
        owner = f"{os.getuid()}:{os.getgid()}"
        for d in missing:
            subprocess.run(["sudo", "chown", owner, d], check=True)
        subprocess.run(["sudo", "chmod", "a+w", path], check=True)


def find_data_paths(orig_dir, freq_labels, date):
    """Find .ms directories and .ms.tar files under {orig_dir}/{freq}MHz/{date}.

    Walks the tree without descending into the measurement sets themselves.
    """
    data_paths = []
    for freq in freq_labels:
        base = f"{orig_dir}/{freq}MHz/{date}"
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            for d in list(dirnames):
                if d.endswith(".ms"):
                    data_paths.append(os.path.join(dirpath, d))
                    dirnames.remove(d)  # Do not descend into the .ms
            for f in filenames:
                if f.endswith(".ms.tar"):
                    data_paths.append(os.path.join(dirpath, f))
    return data_paths


def delete_empty_directories(directory, date, freqs):
    for freq in freqs:
        path = f"{directory}/{freq}MHz/{date}"
        if os.path.isdir(path):
            # -delete is depth-first, so this also removes the (then empty) date dir
            subprocess.run(
                ["sudo", "find", path, "-type", "d", "-empty", "-delete"],
                check=True,
            )


# ---------------------------------------------------------------------------
# Data bookkeeping
# ---------------------------------------------------------------------------
def populate_data_dict(data_paths):
    """Parse filenames into times and frequencies.

    Files that do not match the expected naming are skipped. If the same
    observation exists as both .ms and .ms.tar, the .ms is used; other
    duplicates are ignored (and therefore never deleted).
    """
    entries = {}  # (timestamp, freq) -> (path, is_tar)
    for path in data_paths:
        match = FILENAME_RE.match(os.path.basename(path))
        if match is None:
            print(f"WARNING: Unrecognized filename {path}. Skipping.")
            continue
        key = (match.group(1), match.group(2))
        is_tar = path.endswith(".tar")
        if key in entries:
            existing_path, existing_is_tar = entries[key]
            if existing_is_tar and not is_tar:
                print(f"WARNING: Duplicate for {key}; using {path} over {existing_path}.")
                entries[key] = (path, is_tar)
            else:
                print(f"WARNING: Duplicate for {key}; ignoring {path}.")
            continue
        entries[key] = (path, is_tar)

    keys = sorted(entries.keys())
    times = np.array(
        [datetime.datetime.strptime(k[0], "%Y%m%d_%H%M%S") for k in keys],
        dtype="datetime64[s]",
    )
    return {
        "paths": [entries[k][0] for k in keys],
        "freqs": np.array([k[1] for k in keys]),
        "times": times,
    }


def get_time_inds(data_dict, time_interval_minutes=2, time_integration_s=10):
    """Assign each file to a fixed time chunk.

    Chunks are anchored to midnight of the first file's day (not to the first
    file found), so chunk boundaries and output filenames are stable across
    reruns even after earlier files have been deleted. Uses integer binning,
    so memory use is O(files).
    """
    times = data_dict["times"]
    midnight = np.min(times).astype("datetime64[D]").astype("datetime64[s]")
    secs = (times - midnight).astype("timedelta64[s]").astype(np.int64)
    interval_s = int(round(time_interval_minutes * 60))
    # Equivalent to nearest-center binning with centers offset by (interval - integration) / 2
    data_dict["time_inds"] = (2 * secs + int(time_integration_s)) // (2 * interval_s)
    return data_dict


def get_orig_freq_intervals():
    labels = ["27", "32", "36", "41", "46", "50", "55", "59", "64", "69", "73", "78", "82"]
    start0, step, width = 27179687.5, 4593750.0, 4569824.21875
    return {
        label: np.array([start0 + i * step, start0 + i * step + width])
        for i, label in enumerate(labels)
    }


def get_new_freq_intervals():
    # Determined by where the equalization coefficients are held constant
    return {
        "34": np.array([28352050.78125, 41295898.4375]),
        "44": np.array([41319824.21875, 48354003.90625]),
        "52": np.array([48377929.6875, 56823730.46875]),
        "62": np.array([56847656.25, 67398925.78125]),
        "72": np.array([67422851.5625, 77615234.375]),
        "79": np.array([77639160.15625, 82256835.9375]),
        "83": np.array([82280761.71875, 84649414.0625]),
    }


def get_contributing_freqs(new_freq_interval_dict, orig_freq_interval_dict):
    contributing_freqs = []
    for new_interval in new_freq_interval_dict.values():
        contributing_freqs.append(
            [
                freq
                for freq, orig_interval in orig_freq_interval_dict.items()
                if orig_interval[0] <= new_interval[1]
                and orig_interval[1] >= new_interval[0]
            ]
        )
    return contributing_freqs


def chunk_is_complete(data_dict, use_inds, contributing_freqs):
    """True if every contributing frequency has exactly the expected file count."""
    freqs = data_dict["freqs"][use_inds]
    return all(
        np.count_nonzero(freqs == f) == FILES_PER_FREQ_PER_CHUNK
        for f in contributing_freqs
    )


def get_output_filename(data_dict, use_inds, output_freq, output_dir, is_tmp_dir=False):
    times = data_dict["times"][use_inds]
    tmin = np.min(times).item()
    tmax = np.max(times).item()
    name = f"{tmin.strftime('%Y%m%d_%H%M%S')}-{tmax.strftime('%H%M%S')}_{output_freq}MHz.ms"
    if is_tmp_dir:
        return f"{output_dir}/{name}"
    return f"{output_dir}/{output_freq}MHz/{tmin.strftime('%Y-%m-%d')}/{tmin.strftime('%H')}/{name}"


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------
def copy_files_to_tmp(filepaths, work_dir):
    """Copy (or extract) each input into work_dir. Raises on failure."""
    tmp_filepaths = []
    for src in filepaths:
        name = os.path.basename(src)
        if name.endswith(".tar"):
            # Extract straight from the source, avoiding a copy of the tar
            subprocess.run(["tar", "-xf", src, "-C", work_dir], check=True)
            dest = os.path.join(work_dir, name.removesuffix(".tar"))
        elif os.path.isdir(src):
            dest = os.path.join(work_dir, name)
            shutil.copytree(src, dest)
        else:
            raise RuntimeError(f"Unexpected input type: {src}")

        if not os.path.isdir(dest):
            raise RuntimeError(
                f"Copying {src} to {work_dir} did not produce {dest}. "
                f"Contents: {os.listdir(work_dir)}"
            )
        tmp_filepaths.append(dest)
    return tmp_filepaths


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

    if not refresh and os.path.isdir(output_filename):
        print(f"File {output_filename} already exists. Skipping.")
        return STATUS_EXISTS

    filepaths = [data_dict["paths"][ind] for ind in use_inds]
    frequencies = data_dict["freqs"][use_inds]
    times = data_dict["times"][use_inds]

    work_dir = None  # Private per-call directory; removed in the finally block
    try:
        if tmp_dir is None:
            use_filepaths = filepaths
        else:
            ensure_dir(tmp_dir)
            work_dir = tempfile.mkdtemp(prefix="concat_", dir=tmp_dir)
            use_filepaths = copy_files_to_tmp(filepaths, work_dir)

        lookup = {(t, f): p for t, f, p in zip(times, frequencies, use_filepaths)}
        unique_times = np.unique(times)
        unique_freqs = np.unique(frequencies)

        uv = None
        for time_ind, time in enumerate(unique_times):
            uv_single_freq = None
            for freq_ind, freq in enumerate(unique_freqs):
                path_use = lookup.get((time, freq))
                if path_use is None:
                    print(f"ERROR: No file for time {time} and frequency {freq}MHz.")
                    return STATUS_MISSING

                uv_new = pyuvdata.UVData()
                try:
                    print(f"Reading file {path_use}")
                    uv_new.read(path_use)
                except Exception as e:
                    print(f"WARNING: Error reading file {path_use}: {e!r}. Skipping.")
                    return STATUS_UNREADABLE

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

        # Select the output band, with half-a-channel tolerance at the edges.
        # ravel() handles both the 1D (pyuvdata >= 3) and (1, Nfreqs) shapes.
        freq_array = np.asarray(uv.freq_array).ravel()
        tol = np.median(np.abs(np.diff(freq_array))) / 2 if freq_array.size > 1 else 0.0
        lo, hi = new_freq_interval_dict[output_freq]
        keep_freqs = np.where((freq_array >= lo - tol) & (freq_array <= hi + tol))[0]
        if keep_freqs.size == 0:
            print(f"ERROR: No channels fall within the {output_freq}MHz band.")
            return STATUS_ERROR
        uv.select(frequencies=freq_array[keep_freqs], inplace=True)

        outdir = os.path.dirname(output_filename)
        if tmp_dir is None:
            ensure_dir(outdir)
            uv.write_ms(output_filename, clobber=True)
            if not os.path.isdir(output_filename):
                print(f"Error writing file {output_filename}.")
                return STATUS_ERROR
        else:
            tmp_output = get_output_filename(
                data_dict, use_inds, output_freq, work_dir, is_tmp_dir=True
            )
            uv.write_ms(tmp_output, clobber=True)
            if not os.path.isdir(tmp_output):
                print(f"Error writing file {tmp_output}.")
                return STATUS_ERROR

            ensure_dir(outdir)
            if refresh and os.path.isdir(output_filename):
                subprocess.run(["sudo", "rm", "-r", output_filename], check=True)
            subprocess.run(["mv", tmp_output, outdir], check=True)
            if not os.path.isdir(output_filename):
                print(f"Error moving file to {output_filename}.")
                return STATUS_ERROR

        print(f"New file written to {output_filename}")
        return STATUS_SUCCESS

    except Exception as e:
        print(f"ERROR: Failed to produce {output_filename}: {e!r}")
        return STATUS_ERROR
    finally:
        # Removes the copied inputs and any partial output, however we exit
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_concatenate_data(
    date,
    orig_dir="/lustre/pipeline/cosmology",
    output_dir="/lustre/pipeline/cosmology/concatenated_data",
    tmp_dir="/fast/rbyrne/data_concatenation_tmp",
    delete_files=True,
    refresh=False,
):
    new_freq_interval_dict = get_new_freq_intervals()
    orig_freq_interval_dict = get_orig_freq_intervals()
    contributing_freqs = get_contributing_freqs(
        new_freq_interval_dict, orig_freq_interval_dict
    )

    data_paths = find_data_paths(orig_dir, orig_freq_interval_dict.keys(), date)
    print(f"Found {len(data_paths)} input files.")
    data_dict = populate_data_dict(data_paths)

    if len(data_dict["times"]) == 0:
        print("No files found.")
        if delete_files:
            delete_empty_directories(orig_dir, date, orig_freq_interval_dict.keys())
        return

    data_dict = get_time_inds(data_dict)

    # Which files can contribute to each output band (computed once)
    band_masks = [
        np.isin(data_dict["freqs"], contributing)
        for contributing in contributing_freqs
    ]

    for time_ind in np.unique(data_dict["time_inds"]):  # Chronological order
        file_inds_time_match = np.where(data_dict["time_inds"] == time_ind)[0]
        status = []
        for band_ind, output_freq in enumerate(new_freq_interval_dict.keys()):
            use_inds = file_inds_time_match[band_masks[band_ind][file_inds_time_match]]
            if not chunk_is_complete(data_dict, use_inds, contributing_freqs[band_ind]):
                print(
                    f"ERROR: Not all files are present for time chunk {time_ind} "
                    f"and frequency {output_freq}MHz."
                )
                status.append(STATUS_MISSING)
                continue
            status.append(
                concatenate_files(
                    data_dict,
                    use_inds,
                    output_freq,
                    new_freq_interval_dict,
                    output_dir,
                    tmp_dir=tmp_dir,
                    refresh=refresh,
                )
            )

        # Delete inputs only if every band for this chunk succeeded or already existed
        if delete_files and all(s in DELETE_OK_STATUSES for s in status):
            for file_ind in file_inds_time_match:
                filename = data_dict["paths"][file_ind]
                print(f"Deleting {filename}")
                subprocess.run(["sudo", "rm", "-r", filename], check=True)

    if delete_files:
        delete_empty_directories(orig_dir, date, orig_freq_interval_dict.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate MS files into new bands.")
    parser.add_argument("date", nargs="?", default="2026-10-22")
    parser.add_argument("--delete-files", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run_concatenate_data(
        args.date, delete_files=args.delete_files, refresh=args.refresh
    )
