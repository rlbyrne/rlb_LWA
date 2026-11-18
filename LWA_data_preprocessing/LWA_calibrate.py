from typing import Optional, List
import os
import subprocess
import sys
import shutil
import warnings
import numpy as np
import pyuvdata
import datetime
from generate_model_vis_fftvis import run_fftvis_sim
from generate_model_vis import run_pyuvsim
import smooth_cal_solutions
from calico import calibration_wrappers, calibration_qa


def concatenate_ms_files(
    use_files_full_paths,
    output_filename,
    script_path="/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/concatenate_ms_files.py",
):

    file_list_str = ""
    for filename in use_files_full_paths:
        file_list_str += f"{filename} "
    subprocess.run(
        [
            "python",
            script_path,
            "--path_in",
            file_list_str,
            "--path_out",
            output_filename,
        ],
        check=True,
    )


def get_bad_antenna_list(
    year,
    month,
    day,
    conda_env="/opt/devel/pipeline/envs/development",
    script_path="/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/get_bad_ants.py",
):

    call_str = (
        f"conda run --prefix {conda_env} python {script_path} {year} {month} {day}"
    )
    result = subprocess.getoutput(call_str)
    result = result.split("\n")
    result = [line for line in result if line.startswith("get_bandants output:")][0]
    result = result.split("get_bandants output:")[1].strip()
    result = eval(result)
    if np.isfinite(result[0]):
        return result[1]
    else:
        print(
            f"ERROR: No information on bad antennas found for {year}/{month}/{day}. Exiting."
        )
        sys.exit(1)


def flag_antennas(
    uvd,
    antenna_names=[],
    inplace=False,
):

    # Map telescope antenna names (format "LWA###") to antenna numbers
    tel_ant_nums = dict(
        zip(uvd.telescope.antenna_names, uvd.telescope.antenna_numbers)
    )

    # Collect antenna numbers to flag for each feed
    x_ants = []  # Polarization A
    y_ants = []  # Polarization B
    all_ants = []  # No polarization given: flag all polarizations
    for ant_name in antenna_names:
        ant_name_number_comp = ant_name.strip("LWA").strip("-").strip("A").strip("B")
        ant_num = tel_ant_nums.get(f"LWA{ant_name_number_comp}")
        if ant_num is None:
            warnings.warn(f"Antenna {ant_name} not found in antenna_names. Skipping.")
            continue
        if ant_name.endswith("A"):
            x_ants.append(ant_num)
        elif ant_name.endswith("B"):
            y_ants.append(ant_num)
        else:
            all_ants.append(ant_num)
    x_ants = x_ants + all_ants
    y_ants = y_ants + all_ants

    # Baselines whose first or second antenna is flagged, for each feed
    ant1_x = np.isin(uvd.ant_1_array, x_ants)
    ant2_x = np.isin(uvd.ant_2_array, x_ants)
    ant1_y = np.isin(uvd.ant_1_array, y_ants)
    ant2_y = np.isin(uvd.ant_2_array, y_ants)
    pol_flag_bls = {
        -5: ant1_x | ant2_x,  # XX
        -6: ant1_y | ant2_y,  # YY
        -7: ant1_x | ant2_y,  # XY
        -8: ant1_y | ant2_x,  # YX
    }
    # Any other polarizations are flagged only for antennas with no polarization given
    other_flag_bls = np.isin(uvd.ant_1_array, all_ants) | np.isin(
        uvd.ant_2_array, all_ants
    )

    if inplace:
        uvd_new = uvd
    else:
        uvd_new = uvd.copy()

    for pol_ind, pol in enumerate(uvd_new.polarization_array):
        flag_bls = pol_flag_bls.get(pol, other_flag_bls)
        uvd_new.flag_array[flag_bls, :, pol_ind] = True

    if not inplace:
        return uvd_new


def flag_outriggers(
    uvd,
    core_center_coords=None,
    core_radius_m=130.0,
    remove_outriggers=False,
    inplace=False,
):
    """
    Flag (and optionally remove) all baselines that include an outrigger antenna.

    Parameters
    ----------
    uvd : pyuvdata.UVData
        Data to flag.
    core_center_coords : list of float or None
        [East, North] position of the core center in meters, relative to the
        telescope location. If None, the median antenna position (over all
        antennas in uvd.telescope, including outriggers) is used. Default None.
    core_radius_m : float
        Antennas farther than this horizontal distance from the core center are
        treated as outriggers. Default 130.
    remove_outriggers : bool
        If True, baselines including outriggers are removed from the data rather
        than only flagged. Default False.
    inplace : bool
        If True, modify uvd in place. Otherwise return a modified copy. Default False.

    Returns
    -------
    pyuvdata.UVData or None
        The flagged copy if inplace=False, otherwise None.
    """

    # Antenna positions in topocentric (East, North, Up) coordinates
    antpos = uvd.telescope.get_enu_antpos()

    if core_center_coords is None:
        core_center_coords = [np.median(antpos[:, 0]), np.median(antpos[:, 1])]
    outrigger_ant_inds = np.where(
        np.sqrt(
            (antpos[:, 0] - core_center_coords[0]) ** 2.0
            + (antpos[:, 1] - core_center_coords[1]) ** 2.0
        )
        > core_radius_m
    )[0]
    outrigger_ants = np.asarray(uvd.telescope.antenna_numbers)[outrigger_ant_inds]

    if inplace:
        uvd_new = uvd
    else:
        uvd_new = uvd.copy()

    # Flag all baselines with an outrigger on either end
    flag_bls = np.isin(uvd_new.ant_1_array, outrigger_ants) | np.isin(
        uvd_new.ant_2_array, outrigger_ants
    )
    uvd_new.flag_array[flag_bls, :, :] = True

    if remove_outriggers:
        # Keep only antennas that are present in the data
        data_ants = np.unique(
            np.concatenate((uvd_new.ant_1_array, uvd_new.ant_2_array))
        )
        keep_ants = data_ants[~np.isin(data_ants, outrigger_ants)]
        uvd_new.select(antenna_nums=keep_ants)

    if not inplace:
        return uvd_new


def get_model_visibilities(
    model_visibility_mode=None,
    refresh_model=False,
    model_vis_file=None,
    include_diffuse=False,
    lst_lookup_table="/lustre/21cmpipe/simulation_outputs/lst_lookup_table.csv",
    freq_band=None,
    data_file=None,
    lst_array=None,
    time_array=None,
    skymodel_path=None,
    diffuse_skymodel_path=None,
    beam_path="/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits",
    simulation_package="fftvis",
):
    """
    model_visibility_mode : str
        Options: "read file", "LST interpolate", or "run simulation".
    refresh_model : bool
        If True, model visibilities will be recalculated even if model_vis_file exists.
        Default False.
    model_vis_file : str
        If model_visibility_mode is "read file", path to the saved model visibility
        file in a pyuvdata-readable format. If model_visibility_mode is "LST interpolate"
        or "run simulation", path to an output ms file. Model visibilities will
        be written to this file.
    include_diffuse : bool
        Used only if model_visibility_mode is "LST interpolate" or "run simulation".
        Default False.
    lst_lookup_table : str
        Required and used only if model_visibility_mode is "LST interpolate".
    freq_band : str
        Required and used only if model_visibility_mode is "LST interpolate".
    data_file : str
        Path to a reference pyuvdata-readable data file for metadata. Required if
        model_visibility_mode is "run simulation". Required if model_visibility_mode
        is "LST interpolate" and either lst_array or time_array is None.
    lst_array : array of float
        Required and used only if model_visibility_mode is "LST interpolate".
    time_array : array of float
        Required and used only if model_visibility_mode is "LST interpolate".
    skymodel_path : str
        Required and used only if model_visibility_mode is "run simulation". Path
        to a pyradiosky-formatted sky model file.
    diffuse_skymodel_path : str
        Required and used only if model_visibility_mode is "run simulation" and
        include_diffuse is True. Path to a pyradiosky-formatted sky model file.
    beam_path : str
        Required and used only if model_visibility_mode is "run simulation". Path
        to a pyuvdata-formatted beam fits file.
    simulation_package : str
        Defines what simulation package to use. Used only if model_visibility_mode
        is "run simulation". Options are "pyuvsim" or "fftvis" ("matvis" may be added
        in the future).
    """

    if model_visibility_mode != "read file" and not refresh_model:
        if os.path.isdir(model_vis_file) or os.path.isfile(model_vis_file):
            print(
                f"File {model_vis_file} already exists. Skipping visibility simulation."
            )
            model_visibility_mode = "read file"

    if model_visibility_mode == "read file":
        if not os.path.isdir(model_vis_file) and not os.path.isfile(model_vis_file):
            print(f"ERROR: File {model_vis_file} not found. Exiting.")
            sys.exit(1)

    elif model_visibility_mode == "LST interpolate":
        if os.path.isdir(model_vis_file):
            print(f"ERROR: File {model_vis_file} exits. Exiting.")
            sys.exit(1)
        if not os.path.isdir(os.path.dirname(model_vis_file)):
            print(
                f"ERROR: Directory {os.path.dirname(model_vis_file)} not found. Exiting."
            )
            sys.exit(1)

        if lst_array is None or time_array is None:  # Get metadata
            data_reference = pyuvdata.UVData()
            data_reference.read(data_file, read_data=False)
            if lst_array is None:
                lst_array = data_reference.lst_array
            if time_array is None:
                time_array = data_reference.time_array
            data_reference = None

        model_dir = "/lustre/rbyrne/simulation_outputs"
        with open(lst_lookup_table, "r") as f:
            lst_data = f.readlines()
        model_lsts = np.array([])
        model_lst_filenames = np.array([])
        for line in lst_data[1:]:
            line_split = line.replace("\n", "").strip().split(",")
            model_lsts = np.append(model_lsts, float(line_split[0]))
            model_lst_filenames = np.append(
                model_lst_filenames,
                f"{line_split[1]}_{freq_band}MHz_source_sim.uvfits".strip(),
            )

        model_uv_list = []
        for time_ind, use_lst in enumerate(list(set(lst_array))):
            print(
                f"Calculating model visibilities for time step {time_ind+1} of {len(list(set(lst_array)))}."
            )
            lst_distance = np.abs(model_lsts - use_lst)
            ind1 = np.where(lst_distance == np.min(lst_distance))[0]
            ind2 = np.where(lst_distance == np.sort(lst_distance)[1])[0]
            lst1 = model_lsts[ind1]
            model_filename1 = model_lst_filenames[ind1][0]
            lst2 = model_lsts[ind2]
            model_filename2 = model_lst_filenames[ind2][0]

            # Interpolate models
            model1_uv = pyuvdata.UVData()
            model1_uv.read(f"{model_dir}/{model_filename1}")
            model1_uv.select(lsts=[lst1])
            model1_uv.filename = [""]
            model1_uv_diffuse = pyuvdata.UVData()

            if include_diffuse:
                model1_uv_diffuse.read(
                    f"{model_dir}/{model_filename1.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits"
                )
                model1_uv_diffuse.select(lsts=[lst1])
                model1_uv_diffuse.filename = [""]
                model1_uv.sum_vis(model1_uv_diffuse, inplace=True)

            model2_uv = pyuvdata.UVData()
            model2_uv.read(f"{model_dir}/{model_filename2}")
            model2_uv.select(lsts=[lst2])
            model2_uv.filename = [""]

            if include_diffuse:
                model2_uv_diffuse = pyuvdata.UVData()
                model2_uv_diffuse.read(
                    f"{model_dir}/{model_filename2.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits"
                )
                model2_uv_diffuse.select(lsts=[lst2])
                model2_uv_diffuse.filename = [""]
                model2_uv.sum_vis(model2_uv_diffuse, inplace=True)

            # Phase to consistent phase center
            model1_uv.phase_to_time(np.mean(time_array))
            model2_uv.phase_to_time(np.mean(time_array))

            # Combine data
            model1_uv.data_array *= np.abs(lst1 - use_lst) / np.abs(lst2 - lst1)
            model2_uv.data_array *= np.abs(lst2 - use_lst) / np.abs(lst2 - lst1)
            model_uv = model1_uv.sum_vis(
                model2_uv,
                inplace=False,
                run_check=False,
                check_extra=False,
                run_check_acceptability=False,
                override_params=[
                    "lst_array",
                    "time_array",
                    "uvw_array",
                    "filename",
                ],
            )
            # Correct for decoherence
            model1_uv.data_array = np.abs(model1_uv.data_array) + 0 * 1j
            model2_uv.data_array = np.abs(model2_uv.data_array) + 0 * 1j
            model_uv_abs = model1_uv.sum_vis(
                model2_uv,
                inplace=False,
                run_check=False,
                check_extra=False,
                run_check_acceptability=False,
                override_params=[
                    "lst_array",
                    "time_array",
                    "uvw_array",
                    "filename",
                ],
            )
            model_uv.data_array *= np.abs(model_uv_abs.data_array) / np.abs(
                model_uv.data_array
            )

            model_uv.time_array = np.full(
                model_uv.Nblts, np.sort(list(set(time_array)))[time_ind]
            )
            model_uv.lst_array = np.full(model_uv.Nblts, use_lst)
            model_uv_list.append(model_uv)

        # Combine LSTs
        combined_model_uv = model_uv_list[0]
        if len(model_uv_list) > 1:
            for model_uv_use in model_uv_list[1:]:
                combined_model_uv.fast_concat(model_uv_use, "blt", inplace=True)

        print(f"Saving model file to {model_vis_file}.")
        if model_vis_file.endswith(".ms"):
            combined_model_uv.write_ms(model_vis_file, force_phase=True, fix_autos=True)
        else:
            combined_model_uv.write_uvfits(
                model_vis_file, force_phase=True, fix_autos=True, uvw_double=False
            )

    elif model_visibility_mode == "run simulation":
        if os.path.isdir(model_vis_file):
            print(f"ERROR: File {model_vis_file} exits. Exiting.")
            sys.exit(1)
        if not os.path.isdir(os.path.dirname(model_vis_file)):
            print(
                f"ERROR: Directory {os.path.dirname(model_vis_file)} not found. Exiting."
            )
            sys.exit(1)

        if simulation_package == "pyuvsim":
            simulation_script = run_pyuvsim
        elif simulation_package == "fftvis":
            simulation_script = run_fftvis_sim
        else:
            print(
                f"ERROR: Unknown option for simulation_package {simulation_package}. Exiting."
            )
            sys.exit(1)

        if include_diffuse:
            compact_source_output_filepath = (
                f"{model_vis_file.removesuffix('.ms')}compact.ms"
            )
            diffuse_output_filepath = f"{model_vis_file.removesuffix('.ms')}diffuse.ms"
        else:
            compact_source_output_filepath = model_vis_file
        if not refresh_model and os.path.isdir(compact_source_output_filepath):
            print(
                f"File {compact_source_output_filepath} already exists. Skipping visibility simulation."
            )
        else:
            simulation_script(
                catalog=skymodel_path,
                beam_path=beam_path,
                input_data=data_file,
                output_path=compact_source_output_filepath,
            )
        if include_diffuse:
            if not refresh_model and os.path.isdir(diffuse_output_filepath):
                print(
                    f"File {diffuse_output_filepath} already exists. Skipping visibility simulation."
                )
            else:
                simulation_script(
                    catalog=diffuse_skymodel_path,
                    beam_path=beam_path,
                    input_data=data_file,
                    output_path=diffuse_output_filepath,
                )
            # Combine compact and diffuse models
            compact_sim = pyuvdata.UVData()
            diffuse_sim = pyuvdata.UVData()
            compact_sim.read(compact_source_output_filepath)
            diffuse_sim.read(diffuse_output_filepath)
            compact_sim.sum_vis(diffuse_sim, inplace=True)
            if model_vis_file.endswith(".ms"):
                compact_sim.write_ms(
                    model_vis_file, force_phase=True, fix_autos=True, clobber=True
                )
            else:
                compact_sim.write_uvfits(
                    model_vis_file,
                    force_phase=True,
                    fix_autos=True,
                    uvw_double=False,
                    clobber=True,
                )
            if os.path.isdir(model_vis_file):  # Confirm write was successful
                # Delete intermediate data products
                subprocess.run(["rm", compact_source_output_filepath], check=True)
                subprocess.run(["rm", diffuse_output_filepath], check=True)

    else:
        print(
            f"ERROR: Unknown option for model_visibility_mode {model_visibility_mode}."
        )
        print(
            f'Options are "read file", "LST interpolate", or "run simulation". Exiting.'
        )
        sys.exit(1)


def get_datafiles(
    freq_band="41",
    start_time=datetime.datetime(2025, 5, 8, 16, 7, 36),  # In UTC
    delta_time=datetime.timedelta(minutes=2),  # Total time interval
    time_step=0,  # Offset start time to step through time intervals
    raw_data_dir="/lustre/pipeline/calibration",
    save_dir="/lustre/21cmpipe",
):

    use_start_time = start_time + time_step * delta_time
    end_time = start_time + (time_step + 1) * delta_time
    min_time_str = use_start_time.strftime("%H%M%S")
    max_time_str = end_time.strftime("%H%M%S")
    year = use_start_time.strftime("%Y")
    month = use_start_time.strftime("%m")
    day = use_start_time.strftime("%d")
    hour = use_start_time.strftime("%H")

    datadir = f"{raw_data_dir}/{freq_band}MHz/{year}-{month}-{day}/{hour}"
    output_dir = f"{save_dir}/{year}-{month}-{day}"

    if not os.path.isdir(output_dir):  # Make target directory if it does not exist
        os.mkdir(output_dir)

    all_files = os.listdir(datadir)
    use_files = [
        filename
        for filename in all_files
        if filename.startswith(f"{year}{month}{day}") and filename.endswith(".ms")
    ]
    use_files = [
        filename
        for filename in use_files
        if (int(filename.split("_")[1]) >= int(min_time_str))
        and int(filename.split("_")[1]) < int(max_time_str)
    ]
    if len(use_files) != 12:
        print("ERROR: Number of files found is not 12.")
        sys.exit(1)
    use_files.sort()

    concatenated_filename = f"{year}{month}{day}_{use_files[0].split('_')[1]}-{use_files[-1].split('_')[1]}_{freq_band}MHz.ms"

    if not os.path.isdir(f"{output_dir}/{concatenated_filename}"):

        # Copy files
        for filename in use_files:
            if not os.path.isdir(f"{output_dir}/{filename}"):
                print(f"Copying file {filename}")
                subprocess.run(
                    ["cp", "-r" f"{datadir}/{filename}", f"{output_dir}/{filename}"],
                    check=True,
                )

        if len(use_files) > 1:  # Concatenate files
            use_files_full_paths = [
                f"{output_dir}/{filename}" for filename in use_files
            ]
            print("Concatenating files.")
            concatenate_ms_files(
                use_files_full_paths, f"{output_dir}/{concatenated_filename}"
            )
            if not os.path.isdir(f"{output_dir}/{concatenated_filename}"):
                print("ERROR: Concatenation failed. Exiting.")
                sys.exit()
        else:  # Only one file used
            concatenated_filename = use_files[0]

    return f"{output_dir}/{concatenated_filename}"


def flag_data(
    datafile_path: str,
    run_aoflagger: bool = True,
    flag_antennas_from_autocorrs: bool = True,
    flag_antenna_list: List[str] = [],
):

    # Get antennas to flag based on Andrea's autocorrelation metrics
    if flag_antennas_from_autocorrs:
        output_file_prefix = os.path.splitext(os.path.basename(datafile_path))[0]
        date = datetime.datetime(
            int(output_file_prefix[:4]),
            int(output_file_prefix[4:6]),
            int(output_file_prefix[6:8]),
        )
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        flag_antenna_list_autocorrs = get_bad_antenna_list(year, month, day)
        print(
            f"Using antenna autocorrelation data to flag antennas. Flagging {flag_antenna_list_autocorrs}."
        )
        flag_antenna_list = np.unique(
            np.concatenate((flag_antenna_list, flag_antenna_list_autocorrs))
        )

    if len(flag_antenna_list) > 0:
        uv = pyuvdata.UVData()
        uv.read(datafile_path, data_column="DATA")
        uv.phase_to_time(np.mean(uv.time_array))
        flag_antennas(
            uv,
            antenna_names=flag_antenna_list,
            inplace=True,
        )
        uv.write_ms(datafile_path, fix_autos=True, clobber=True)
        uv = None  # Clear memory

    if run_aoflagger:
        subprocess.run(["aoflagger", datafile_path], check=True)


def read_caltable(input_path, use_freqs=None):

    cal = pyuvdata.UVCal()
    cal.read(input_path)

    if os.path.isdir(input_path):  # Check if CASA caltable
        # Collapse spws
        gain = np.ma.masked_where(cal.flag_array, cal.gain_array)
        gain = np.ma.sum(gain, axis=2)
        if use_freqs is None:
            freq_inds = np.arange(cal.Nfreqs)
            use_freqs = cal.freq_array
        else:
            freq_inds = np.digitize(use_freqs, cal.freq_array, right=True)
        gain = gain[:, freq_inds, :]
        gain.shape = (cal.Nants_data, len(freq_inds), 1, cal.Njones)

        # problem. we cant combine flags because the time staggering means that every time is flagged at least once
        flags = np.zeros(gain.shape, dtype=bool)
        flags[np.abs(gain) > 0.999] = 1  # in my file, gain=1 was flagged.
        flags[np.abs(gain) < 1e-9] = 1  # no gain no pain

        cal.select(times=cal.time_array[-1], frequencies=use_freqs)
        cal.flex_spw_id_array = np.zeros_like(use_freqs, dtype=int)
        cal.spw_array = np.array([0])
        cal.Nspws = 1
        cal.gain_array = np.array(gain)
        cal.flag_array = np.array(flags)

    return cal


def peel_sources(
    out_ms_path,
    orig_ms_path,
    julia_path="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin/julia",
    ttcal_path="/opt/devel/nkosogor/nkosogor/TTCalX",  # TTCalX install directory
    peel_sources_path="/home/pipeline/sources.json",
    peel_mode="peel",  # Options include "peel", "zest", "shave", "prune"
    maxiter=5,
    tolerance=1e-4,
    minuvw=10,
    Ntimes=None,
    column="DATA",
    use_gpu=True,
) -> None:

    if Ntimes != 1:
        uv = pyuvdata.UVData()
        uv.read(orig_ms_path)
        Ntimes = uv.Ntimes

    ms_file_list = []
    if Ntimes != 1:  # Need to split file into individual times
        time_array = list(set(uv.time_array))
        for time_ind, time in enumerate(time_array):
            filepath = f"{out_ms_path.strip('.ms')}_time{time_ind}.ms"
            uv_single_time = uv.select(times=time, inplace=False)
            uv_single_time.phase_to_time(time)
            uv_single_time.write_ms(filepath, clobber=True, fix_autos=True)
            ms_file_list.append(filepath)
    else:
        if out_ms_path != orig_ms_path:
            if os.path.isdir(out_ms_path):  # Replace ms file
                subprocess.run(["sudo", "rm", "-rf", out_ms_path], check=True)
            subprocess.run(
                ["cp", "-r", orig_ms_path, out_ms_path], check=True
            )  # Initialized peeled file
        ms_file_list.append(out_ms_path)

    # Run TTCalX (https://github.com/nkosogor/TTCalX) on all files in one call
    # to pay the Julia compilation cost only once
    julia_env = os.environ.copy()
    julia_env["JULIA_DEPOT_PATH"] = (
        f"/tmp/julia_{os.environ.get('USER', 'user')}:/opt/devel/nkosogor/nkosogor/julia_depot"
    )
    if use_gpu:
        ttcal_args = [
            julia_path,
            f"--project={ttcal_path}",
            f"{ttcal_path}/bin/ttcal_gpu.jl",
            peel_mode,
        ]
    else:  # CPU mode uses zest_sky_rfi.jl, which uses a different solver
        if peel_mode not in ["peel", "zest"]:
            raise ValueError("CPU peeling supports only peel_mode 'peel' or 'zest'.")
        ttcal_args = [
            julia_path,
            "-t",
            "auto",
            f"--project={ttcal_path}",
            f"{ttcal_path}/bin/zest_sky_rfi.jl",
            "--cpu",
        ]
        if peel_mode == "peel":
            ttcal_args.append("--peel")
    subprocess.run(
        [
            *ttcal_args,
            f"--maxiter={maxiter}",
            f"--tolerance={tolerance}",
            f"--minuvw={minuvw}",
            f"--column={column}",
            peel_sources_path,
            *ms_file_list,
        ],
        check=True,
        env=julia_env,
    )

    if Ntimes != 1:
        for file_ind, filepath in enumerate(ms_file_list):
            uv_new = pyuvdata.UVData()
            uv_new.read(filepath)
            if file_ind == 0:
                uv = uv_new
            else:
                uv.fast_concat(uv_new, "blt", inplace=True)
        uv.phase_to_time(np.mean(uv.time_array))
        uv.write_ms(out_ms_path, clobber=True, fix_autos=True)

        # Delete temporary files
        if os.path.isdir(out_ms_path):
            for filepath in ms_file_list:
                subprocess.run(["rm", "-rf", filepath], check=True)
        else:
            print(f"ERROR: File {out_ms_path} not written.")


def run_aoflagger_on_file(datafile_path: str, strategy_file: Optional[str] = None):
    """
    Run AOFlagger on a measurement set. Modifies the file in place.

    Parameters
    ----------
    datafile_path : str
        Path to the .ms file.
    strategy_file : str or None
        Path to an AOFlagger strategy file. If None, the default strategy is used.
    """
    if strategy_file is None:
        subprocess.run(["aoflagger", datafile_path], check=True)
    else:
        subprocess.run(["aoquality", "remove", datafile_path], check=True)
        subprocess.run(["aoquality", "collect", datafile_path], check=True)
        subprocess.run(
            ["aoflagger", "--strategy", strategy_file, datafile_path],
            check=True,
        )


def image_data(
    name,
    ms_filepath,
    wsclean_script="/opt/bin/wsclean",
    pol="I",
    niter=0,
    weight="briggs 0",
    multiscale_scale_bias=0.8,
    size=4096,
    scale=0.03125,
    mgain=0.95,
    horizon_mask="10deg",
    auto_threshold=0.5,
    auto_mask=3,
    taper_inner_tukey=30,
    mem=30,
):

    if niter > 0:  # Save source list
        subprocess.run(
            [
                wsclean_script,
                "-pol",
                pol,
                "-multiscale",
                "-multiscale-scale-bias",
                str(multiscale_scale_bias),
                "-size",
                str(size),
                str(size),
                "-scale",
                str(scale),
                "-niter",
                str(niter),
                "-mgain",
                str(mgain),
                "-weight",
                *weight.split(),
                "-no-update-model-required",
                "-mem",
                str(mem),
                "-horizon-mask",
                horizon_mask,
                "-auto-threshold",
                str(auto_threshold),
                "-auto-mask",
                str(auto_mask),
                "-taper-inner-tukey",
                str(taper_inner_tukey),
                "-local-rms",
                "-no-reorder",
                "-save-source-list",
                "-name",
                name,
                ms_filepath,
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                wsclean_script,
                "-pol",
                pol,
                "-multiscale",
                "-multiscale-scale-bias",
                str(multiscale_scale_bias),
                "-size",
                str(size),
                str(size),
                "-scale",
                str(scale),
                "-niter",
                str(niter),
                "-mgain",
                str(mgain),
                "-weight",
                *weight.split(),
                "-no-update-model-required",
                "-mem",
                str(mem),
                "-horizon-mask",
                horizon_mask,
                "-auto-threshold",
                str(auto_threshold),
                "-auto-mask",
                str(auto_mask),
                "-taper-inner-tukey",
                str(taper_inner_tukey),
                "-local-rms",
                "-no-reorder",
                "-name",
                name,
                ms_filepath,
            ],
            check=True,
        )


def calibration_pipeline(
    datafile_path: str,
    output_dir: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    cal_trial_name: Optional[str] = None,
    beam_path: Optional[str] = "/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits",
    skymodel_path: Optional[
        str
    ] = "/lustre/rbyrne/skymodels/Gregg_20250519_source_models.skyh5",
    include_diffuse: bool = False,
    diffuse_skymodel_path: Optional[str] = None,
    apply_cal_path: Optional[str] = None,
    date: Optional[datetime.datetime] = None,
    run_aoflagger: bool = True,
    aoflagger_strategy_file: Optional[
        str
    ] = "/lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua",
    flag_antennas_from_autocorrs: bool = True,
    flag_antenna_list: List[str] = [],
    refresh_flags: bool = False,
    refresh_model: bool = False,
    refresh_calibration: bool = False,
    simulation_package: str = "fftvis",
    calibrate_with_casa: bool = False,
    casa_calibrate_script_path: str = "/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/casa_calibrate.py",
    min_cal_baseline_lambda: Optional[int] = 10,
    max_cal_baseline_lambda: Optional[int] = 125,
    flip_gain_conj: bool = False,
    plot_gains: bool = False,
    apply_calibration: bool = True,
    smooth_cal: bool = False,
    plot_images: bool = True,
    peel: bool = False,
    julia_path: str = "/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin/julia",
    ttcal_path: str = "/opt/devel/nkosogor/nkosogor/TTCalX",
    peel_sources_path: str = "/home/pipeline/sources.json",
    reflag: bool = False,
    peel_use_gpu: bool = True,
) -> None:
    """
    Parameters
    ----------
    datafile_path : str
        Full path to .ms file containing the data.
    output_dir : str or None
        Path to a directory where to calibration outputs will be saved. If None,
        the directory containing datafile_path will be used. Default None.
    tmp_dir : str or None
        Path to a temporary directory for data processing.
    cal_trial_name : str
        Optional string to append to script outputs. Used to distinguish different
        calibration trials on the same dataset. Default None.
    beam_path : str or None
        Path to a pyuvdata-readable beam model, in .fits format. Default
        /lustre/rbyrne/LWA_10to100_MROsoil_efields.fits. Not used if apply_cal_path
        is not None.
    skymodel_path : str or None
        Path to a pyradiosky-readable sky model, in .skyh5 format. Default
        /lustre/rbyrne/skymodels/Gregg_20250519_source_models.skyh5. Not used if
        apply_cal_path is not None.
    include_diffuse : bool
        If True, a Healpix map will be used to model diffuse emission. Default False.
        Not used if apply_cal_path is not None.
    diffuse_skymodel_path : str or None
        Path to a pyradiosky-readable diffuse sky model, in .skyh5 format. Not used if
        apply_cal_path is not None or include_diffuse is False.
    apply_cal_path : str or None
        If not None, do not perform calibration and instead restore and apply a
        saved calibration solution. Path to a pyuvdata-readable calibration file.
    date : datetime object or None
        Date of data. Used for antenna flag lookup and defining the output
        directory. If None, the filename will be parsed, assuming that the
        data filename begins with format YYYYMMDD. Default None.
    run_aoflagger : bool
        If True, AOFlagger will be run on the data file. Note that this modifies
        the data file. Default True.
    aoflagger_strategy_file : str or None
        Path to aoflagger strategy file. Default
        /lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua. If None, default strategy will
        be used. Used only if run_aoflagger=True.
    flag_antennas_from_autocorrs : bool
        Use pre-computed autocorrelation metrics to look up a table of antennas
        to flag. Default True.
    flag_antenna_list : list of str
        List of antennas to flag. String format is "LWA-###A" for antenna ### and
        polarization A. For example, to flag both polarizations of antenna 5, use
        ["LWA-005A", "LWA-005B"]. If flag_antennas_from_autocorrs=True, these
        antennas will be flagged in addition to those in the flagging lookup table.
    refresh_flags : bool
        If True, pre-existing flags in the data file are cleared. Default False.
    refresh_model : bool
        If True, the model visibilities will be re-generated. Used only if apply_cal_path is
        None. Default False.
    refresh_calibration : bool
        If False, an existing .calfits file will be read and applied. If True, the calibration
        will be re-generated. Used only if apply_cal_path is None. Default False.
    simulation_package : str
        Defines what simulation package is used for visibility model. Options are "fftvis" or
        "pyuvsim". Default "fftvis".
    calibrate_with_casa : bool
        If True, use CASA bandpass calibration. If False, use Calico. Default False.
        If CASA bandpass calibration is used, the data file will be modified by applying
        flags and adding the model visibilities to the MODEL column of the .ms file.
    casa_calibrate_script_path : str
        Used only if calibrate_with_casa is True. Path to the script to perform calibration
        with CASA. Default /opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/casa_calibrate.py.
    min_cal_baseline_lambda : int
        Minimum baseline length, in units of wavelengths, to use in calibration. Default
        10. Not used if apply_cal_path is not None.
    max_cal_baseline_lambda : int
        Maximum baseline length, in units of wavelengths, to use in calibration. Default
        125. Not used if apply_cal_path is not None.
    flip_gain_conj : bool
        If True, conjugate gains before applying. This is needed for CASA-generated calibration
        solutions.
    plot_gains : bool
        If True, generate plot of the gains. Plots will be written to a directory gain_plots.
        Default False.
    apply_calibration : bool
        If True, calibration will be applied to the data, and the calibrated
        visibilities will be written to a new file with suffix *_calibrated.ms.
        If model visibilities have been created, residual (calibrated data minus model)
        visibilities will also be computed and written to a file with suffix *_res.ms.
        Default True.
    smooth_cal : bool
        If True, calibration solutions will be smoothed before application. The gain amplitudes
        will be fit with a second order polynomial and the gain phases will be fit with a linear
        function. Default False.
    plot_images : bool
        If True, images will be generated with WSClean. If apply_calibration=True,
        images will be generated of the calibrated data, model, and residual (data
        minus model). If apply_calibration=False, images will be generated of the
        model only. Default True.
    peel : bool
        If True, images will be peeled with TTCalX. Requires that apply_calibration=True.
    julia_path: str
        Path to julia installation. Used only if peel=True. Default
        "/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin/julia".
    ttcal_path : str
        Path to the TTCalX install directory. Used only if peel=True. Default
        "/opt/devel/nkosogor/nkosogor/TTCalX".
    peel_sources_path : str
        Path to sources for peeling. Used only if peel=True. Default "/home/pipeline/sources.json".
    reflag : bool
        If True, AOFlagger is run again after peeling (on the peeled file) or,
        if peel=False, after calibration is applied (on the calibrated file).
        Uses aoflagger_strategy_file. Requires that apply_calibration=True.
        Default False.
    peel_use_gpu : bool
        If True, peel on a GPU with TTCalX's ttcal_gpu.jl. If False, peel on CPU
        threads with TTCalX's zest_sky_rfi.jl, which uses a different solver and
        supports only peel_mode "peel" or "zest". Used only if peel=True. Default True.
    """

    if peel and not apply_calibration:
        warnings.warn("Peeling requires apply_calibration=True. Setting peel=False.")
        peel = False
    if reflag and not apply_calibration:
        warnings.warn(
            "Reflagging requires apply_calibration=True. Setting reflag=False."
        )
        reflag = False

    # Define output filenames
    output_file_prefix = os.path.splitext(os.path.basename(datafile_path))[0]
    if cal_trial_name is not None:
        output_file_prefix = f"{output_file_prefix}_{cal_trial_name}"
    model_filename = f"{output_file_prefix}_source_sim.ms"
    calfits_filename = f"{output_file_prefix}.calfits"
    calibration_log = f"{output_file_prefix}_cal_log.txt"
    gain_plot_dir = "gain_plots/"
    gain_plot_prefix = output_file_prefix
    calibrated_data_ms = f"{output_file_prefix}_calibrated.ms"
    peeled_ms = f"{output_file_prefix}_peeled.ms"
    res_ms = f"{output_file_prefix}_res.ms"
    calibrated_data_image = f"{output_file_prefix}_calibrated"
    model_image = f"{output_file_prefix}_model"
    peeled_image = f"{output_file_prefix}_peeled"
    res_image = f"{output_file_prefix}_res"

    if output_dir is None:  # Use same directory as the data
        output_dir = os.path.dirname(datafile_path)
    if not os.path.isdir(output_dir):  # Make directory if it doesn't already exist
        os.mkdir(output_dir)
    if tmp_dir is not None:
        if tmp_dir == output_dir:
            tmp_dir = None
    if tmp_dir is None:
        use_output_dir = output_dir
        use_datafile_path = datafile_path
    else:
        use_output_dir = f"{tmp_dir}/{output_file_prefix}_tmp_dir"
        if not os.path.isdir(
            use_output_dir
        ):  # Make directory if it doesn't already exist
            os.mkdir(use_output_dir)
        use_datafile_path = f"{use_output_dir}/{os.path.basename(datafile_path)}"
        subprocess.run(
            ["cp", "-r", datafile_path, os.path.dirname(use_datafile_path)], check=True
        )

    if not refresh_calibration and apply_cal_path is None:
        # Check if a .calfits file already exists and restore it if it does
        if os.path.isfile(f"{use_output_dir}/{calfits_filename}"):
            apply_cal_path = f"{use_output_dir}/{calfits_filename}"
        elif tmp_dir is not None:
            if os.path.isfile(f"{output_dir}/{calfits_filename}"):
                subprocess.run(
                    ["cp", f"{output_dir}/{calfits_filename}", use_output_dir],
                    check=True,
                )
                apply_cal_path = f"{use_output_dir}/{calfits_filename}"

    # Get antennas to flag based on Andrea's autocorrelation metrics
    if flag_antennas_from_autocorrs:
        if date is None:  # Attempt to parse the filename to get date
            date = datetime.datetime(
                int(output_file_prefix[:4]),
                int(output_file_prefix[4:6]),
                int(output_file_prefix[6:8]),
            )
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        flag_antenna_list_autocorrs = get_bad_antenna_list(year, month, day)
        print(
            f"Using antenna autocorrelation data to flag antennas. Flagging {flag_antenna_list_autocorrs}."
        )
        flag_antenna_list = np.unique(
            np.concatenate((flag_antenna_list, flag_antenna_list_autocorrs))
        )

    if refresh_flags:
        uv = pyuvdata.UVData()
        uv.read(use_datafile_path)
        uv.flag_array[:, :, :] = False
        uv.write_ms(use_datafile_path, clobber=True, fix_autos=True)
        uv = None

    if run_aoflagger:
        run_aoflagger_on_file(use_datafile_path, strategy_file=aoflagger_strategy_file)

    if apply_cal_path is not None:  # Restore calibration solution
        print(f"Restoring calibration table {apply_cal_path}")
        uvcal = read_caltable(apply_cal_path)
        uv = None  # Define variable

    else:  # Run calibration

        if not refresh_model and os.path.isdir(f"{output_dir}/{model_filename}"):
            print(f"Model file exists. Using {output_dir}/{model_filename}.")
            if tmp_dir is not None:
                subprocess.run(
                    ["cp", "-r", f"{output_dir}/{model_filename}", use_output_dir],
                    check=True,
                )
        else:
            get_model_visibilities(
                model_visibility_mode="run simulation",
                refresh_model=refresh_model,
                model_vis_file=f"{use_output_dir}/{model_filename}",
                include_diffuse=include_diffuse,
                data_file=use_datafile_path,
                skymodel_path=skymodel_path,
                diffuse_skymodel_path=diffuse_skymodel_path,
                beam_path=beam_path,
                simulation_package=simulation_package,
            )

        if calibrate_with_casa:

            # Flag antennas
            # This would be faster with CASA flagging tools but would require converting antenna
            # names to correlator numbers
            if len(flag_antenna_list) > 0:
                uv = pyuvdata.UVData()
                uv.read(use_datafile_path, data_column="DATA")
                # uv.set_uvws_from_antenna_positions(update_vis=False)
                uv.phase_to_time(np.mean(uv.time_array))
                flag_antennas(
                    uv,
                    antenna_names=flag_antenna_list,
                    inplace=True,
                )
                uv.write_ms(use_datafile_path, fix_autos=True, clobber=True)
                uv = None  # Clear memory

            subprocess.run(
                [
                    "python",
                    casa_calibrate_script_path,
                    "--data_file",
                    use_datafile_path,
                    "--model_file",
                    f"{use_output_dir}/{model_filename}",
                    "--min_cal_baseline_lambda",
                    str(min_cal_baseline_lambda),
                    "--max_cal_baseline_lambda",
                    str(max_cal_baseline_lambda),
                ],
                check=True,
            )

            # Convert CASA calibration solutions to .calfits
            uvcal = pyuvdata.UVCal()
            uvcal.read_ms_cal(f"{use_datafile_path.removesuffix('.ms')}.bcal")
            uvcal.write_calfits(f"{use_output_dir}/{calfits_filename}", clobber=True)

        else:  # Calibrate with Calico

            # Read data
            uv = pyuvdata.UVData()
            print(f"Reading file {use_datafile_path}.")
            uv.read(use_datafile_path, data_column="DATA")
            # uv.set_uvws_from_antenna_positions(update_vis=False)
            uv.phase_to_time(np.mean(uv.time_array))

            # Read model
            model_uv = pyuvdata.UVData()
            print(f"Reading file {use_output_dir}/{model_filename}.")
            model_uv.read(f"{use_output_dir}/{model_filename}", data_column="DATA")
            # model_uv.set_uvws_from_antenna_positions(update_vis=False)
            model_uv.phase_to_time(np.mean(uv.time_array))

            if len(flag_antenna_list) > 0:
                flag_antennas(
                    uv,
                    antenna_names=flag_antenna_list,
                    inplace=True,
                )

            # Calibrate
            uvcal = calibration_wrappers.sky_based_calibration_wrapper(
                uv,
                model_uv,
                min_cal_baseline_lambda=min_cal_baseline_lambda,
                max_cal_baseline_lambda=max_cal_baseline_lambda,
                gains_multiply_model=True,
                verbose=True,
                get_crosspol_phase=False,
                log_file_path=f"{use_output_dir}/{calibration_log}",
                xtol=1e-6,
                maxiter=200,
                antenna_flagging_iterations=0,
                parallel=False,
                lambda_val=0,
            )
            print(f"Writing output to {calfits_filename}")
            uvcal.write_calfits(f"{use_output_dir}/{calfits_filename}", clobber=True)

    if plot_gains:
        if not os.path.isdir(f"{use_output_dir}/{gain_plot_dir}"):
            os.mkdir(f"{use_output_dir}/{gain_plot_dir}")
        calibration_qa.plot_gains(
            uvcal,
            plot_output_dir=f"{use_output_dir}/{gain_plot_dir}",
            cal_name=gain_plot_prefix,
            plot_reciprocal=False,
            ymin=0,
            ymax=None,
            zero_mean_phase=False,
        )

    if apply_calibration:

        # Read data
        uv = pyuvdata.UVData()
        uv.read(use_datafile_path, data_column="DATA")
        uv.phase_to_time(np.mean(uv.time_array))

        if (apply_cal_path is not None) and len(
            flag_antenna_list
        ) > 0:  # Need to flag antennas
            flag_antennas(
                uv,
                antenna_names=flag_antenna_list,
                inplace=True,
            )

        if smooth_cal:
            uvcal = smooth_cal_solutions.smooth_cal(
                uvcal, amp_deg=2, phase_deg=1, freq_array_hz=uv.freq_array
            )

        pyuvdata.utils.uvcalibrate(
            uv, uvcal, inplace=True, time_check=False, flip_gain_conj=flip_gain_conj
        )
        uv.write_ms(
            f"{use_output_dir}/{calibrated_data_ms}", fix_autos=True, clobber=True
        )

        if peel:
            peel_sources(
                f"{use_output_dir}/{peeled_ms}",
                f"{use_output_dir}/{calibrated_data_ms}",
                julia_path=julia_path,
                ttcal_path=ttcal_path,
                peel_sources_path=peel_sources_path,
                Ntimes=uv.Ntimes,
                use_gpu=peel_use_gpu,
            )

        if (
            reflag
        ):  # Rerun AOFlagger on the peeled data, or calibrated data if not peeled
            if peel:
                reflag_ms = f"{use_output_dir}/{peeled_ms}"
            else:
                reflag_ms = f"{use_output_dir}/{calibrated_data_ms}"
            print(f"Reflagging {reflag_ms} with AOFlagger.")
            run_aoflagger_on_file(reflag_ms, strategy_file=aoflagger_strategy_file)

        if apply_cal_path is None:  # Subtract model from data

            # Read model file
            model_uv = pyuvdata.UVData()
            print(f"Reading file {use_output_dir}/{model_filename}.")
            model_uv.read(f"{use_output_dir}/{model_filename}", data_column="DATA")
            # model_uv.set_uvws_from_antenna_positions(update_vis=False)
            model_uv.phase_to_time(np.mean(uv.time_array))

            uv.filename = [""]
            model_uv.filename = [""]
            uv.sum_vis(
                model_uv,
                difference=True,
                inplace=True,
                run_check=False,
                check_extra=False,
                override_params=[
                    "scan_number_array",
                    "phase_center_id_array",
                    "telescope",
                    "phase_center_catalog",
                    "filename",
                    "phase_center_app_dec",
                    "nsample_array",
                    "integration_time",
                    "phase_center_frame_pa",
                    "flag_array",
                    "uvw_array",
                    "lst_array",
                    "phase_center_app_ra",
                    "dut1",
                    "earth_omega",
                    "gst0",
                    "rdate",
                    "time_array",
                    "timesys",
                ],
            )
            uv.write_ms(f"{use_output_dir}/{res_ms}", fix_autos=True, clobber=True)

    if plot_images:
        if apply_cal_path is None:  # Plot model
            image_data(
                f"{use_output_dir}/{model_image}",
                f"{use_output_dir}/{model_filename}",
                niter=5000,
            )
        if apply_calibration:  # Plot calibrated data
            image_data(
                f"{use_output_dir}/{calibrated_data_image}",
                f"{use_output_dir}/{calibrated_data_ms}",
                niter=5000,
            )
        if peel:  # Plot peeled data
            image_data(
                f"{use_output_dir}/{peeled_image}",
                f"{use_output_dir}/{peeled_ms}",
                niter=5000,
            )
        if apply_calibration and apply_cal_path is None:  # Plot residual
            image_data(
                f"{use_output_dir}/{res_image}",
                f"{use_output_dir}/{res_ms}",
                niter=5000,
            )

    if tmp_dir is not None:  # Move outputs

        subprocess.run(
            ["rsync", "-a", "--remove-source-files", f"{use_output_dir}/", output_dir],
            check=True,
        )
        subprocess.run(
            ["find", use_output_dir, "-type", "d", "-empty", "-delete"], check=True
        )
