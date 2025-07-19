import os
import subprocess
import ast
import sys
import numpy as np
import pyuvdata
import datetime
from generate_model_vis_fftvis import run_fftvis_sim
from calico import calibration_wrappers


def concatenate_ms_files(
    use_files_full_paths,
    output_filename,
    script_path="/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/concatenate_ms_files.py",
):

    file_list_str = ""
    for filename in use_files_full_paths:
        file_list_str += f"{filename} "
    os.system(
        f"python {script_path} --path_in {file_list_str} --path_out {output_filename}"
    )


def get_bad_antenna_list(
    year,
    month,
    day,
    conda_env="/opt/devel/pipeline/envs/deployment",
    script_path="/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/get_bad_ants.py",
):

    call_str = (
        f"conda run --prefix {conda_env} python {script_path} {year} {month} {day}"
    )
    result = subprocess.getoutput(call_str)
    result = result.split("\n")
    result = [line for line in result if line.startswith("get_bandants output:")][0]
    result = result.split("get_bandants output:")[1].strip()
    result = ast.literal_eval(result)
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

    if inplace:
        flag_arr = uvd.flag_array
    else:
        flag_arr = np.copy(uvd.flag_array)

    for ant_name in antenna_names:
        ant_name_number_comp = ant_name.strip("LWA").strip("-").strip("A").strip("B")
        ant_ind = np.where(
            np.array(uvd.telescope.antenna_names) == f"LWA{ant_name_number_comp}"
        )[
            0
        ]  # uvd.telescope.antenna_names format is "LWA###"
        if np.size(ant_ind) == 0:
            print(f"WARNING: Antenna {ant_name} not found in antenna_names.")
            continue
        flag_bls_1 = np.where(uvd.ant_1_array == ant_ind)[0]
        flag_bls_2 = np.where(uvd.ant_2_array == ant_ind)[0]
        if ant_name.endswith("A"):
            for pol in [-5, -7]:  # XX, XY
                flag_pol_ind = np.where(uvd.polarization_array == pol)[0]
                if len(flag_pol_ind) > 0:
                    flag_arr[flag_bls_1, :, flag_pol_ind] = True
            for pol in [-5, -8]:  # XX, YX
                flag_pol_ind = np.where(uvd.polarization_array == pol)[0]
                if len(flag_pol_ind) > 0:
                    flag_arr[flag_bls_2, :, flag_pol_ind] = True
        elif ant_name.endswith("B"):
            for pol in [-6, -8]:  # YY, YX
                flag_pol_ind = np.where(uvd.polarization_array == pol)[0]
                if len(flag_pol_ind) > 0:
                    flag_arr[flag_bls_1, :, flag_pol_ind] = True
            for pol in [-6, -7]:  # YY, XY
                flag_pol_ind = np.where(uvd.polarization_array == pol)[0]
                if len(flag_pol_ind) > 0:
                    flag_arr[flag_bls_2, :, flag_pol_ind] = True
        else:  # Flag all polarizations
            flag_bls = np.unique(np.concatenate((flag_bls_1, flag_bls_2)))
            flag_arr[flag_bls, :, :] = True

    if inplace:
        uvd.flag_array = flag_arr
    else:
        uvd_new = uvd.copy()
        uvd_new.flag_array = flag_arr
        return uvd_new


def get_model_visibilities(
    model_visilibility_mode=None,
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
):
    """
    model_visilibility_mode : str
        Options: "read file", "LST interpolate", or "run simulation".
    model_vis_file : str
        If model_visilibility_mode is "read file", path to the saved model visibility
        file in a pyuvdata-readable format. If model_visilibility_mode is "LST interpolate"
        or "run simulation", path to an output ms file. Model visibilities will
        be written to this file.
    include_diffuse : bool
        Used only if model_visilibility_mode is "LST interpolate" or "run simulation".
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
    """

    if model_visilibility_mode == "read file":
        if not os.path.isdir(model_vis_file) and not os.path.isfile(model_vis_file):
            print(f"ERROR: File {model_vis_file} not found. Exiting.")
            sys.exit(1)

    elif model_visilibility_mode == "LST interpolate":
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
                model_vis_file, force_phase=True, fix_autos=True
            )

    elif model_visilibility_mode == "run simulation":
        if os.path.isdir(model_vis_file):
            print(f"ERROR: File {model_vis_file} exits. Exiting.")
            sys.exit(1)
        if not os.path.isdir(os.path.dirname(model_vis_file)):
            print(
                f"ERROR: Directory {os.path.dirname(model_vis_file)} not found. Exiting."
            )
            sys.exit(1)

        if include_diffuse:
            compact_source_output_filepath = (
                f"{model_vis_file.removesuffix('.ms')}compact.ms"
            )
            diffuse_output_filepath = f"{model_vis_file.removesuffix('.ms')}diffuse.ms"
        else:
            compact_source_output_filepath = model_vis_file
        run_fftvis_sim(
            map_path=skymodel_path,
            beam_path=beam_path,
            input_data_path=data_file,
            output_path=compact_source_output_filepath,
            log_path=None,
        )
        if include_diffuse:
            run_fftvis_sim(
                map_path=diffuse_skymodel_path,
                beam_path=beam_path,
                input_data_path=data_file,
                output_path=diffuse_output_filepath,
                log_path=None,
            )
            # Combine compact and diffuse models
            compact_sim = pyuvdata.UVData()
            diffuse_sim = pyuvdata.UVData()
            compact_sim.read(compact_source_output_filepath)
            diffuse_sim.read(diffuse_output_filepath)
            compact_sim.sum_vis(diffuse_sim, inplace=True)
            compact_sim.write_ms(model_vis_file, clobber=True, fix_autos=True)
            if model_vis_file.endswith(".ms"):
                compact_sim.write_ms(model_vis_file, force_phase=True, fix_autos=True)
            else:
                compact_sim.write_uvfits(
                    model_vis_file, force_phase=True, fix_autos=True
                )
            if os.path.isdir(model_vis_file):  # Confirm write was successful
                # Delete intermediate data products
                os.system(f"rm {compact_source_output_filepath}")
                os.system(f"rm {diffuse_output_filepath}")

    else:
        print(
            f"ERROR: Unknown option for model_visilibility_mode {model_visilibility_mode}."
        )
        print(
            f'Options are "read file", "LST interpolate", or "run simulation". Exiting.'
        )
        sys.exit(1)


def get_datafiles(
    freq_band="41",
    start_time=datetime.datetime(2025, 5, 8, 16, 7, 36),
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
    concatenated_filename = f"{year}{month}{day}_{use_files[0].split('_')[1]}-{use_files[-1].split('_')[1]}_{freq_band}MHz.ms"

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

    if not os.path.isdir(f"{output_dir}/{concatenated_filename}"):

        # Copy files
        for filename in use_files:
            if not os.path.isdir(f"{output_dir}/{filename}"):
                print(f"Copying file {filename}")
                os.system(f"cp -r {datadir}/{filename} {output_dir}/{filename}")

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

    return concatenated_filename


def calibration_pipeline(
    datafile_path,
    output_dir=None,  # If None, datafile_path directory will be used.
    beam_path="/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits",
    skymodel_path="/lustre/rbyrne/skymodels/Gregg_20250519_source_models.skyh5",
    date=None,  # datetime object. Used to get antenna flags and define output directory. If None, filename will be parsed assuming the filename begins with format YYYYMMDD
    run_aoflagger=True,  # Will modify datafile
):

    # Define output filenames
    output_file_prefix = os.path.splitext(os.path.basename(datafile_path))[0]
    model_filename = f"{output_file_prefix}_source_sim.ms"
    calfits_filename = f"{output_file_prefix}.calfits"
    calibration_log = f"{output_file_prefix}_cal_log.txt"
    calibrated_data_ms = f"{output_file_prefix}_calibrated.ms"
    res_ms = f"{output_file_prefix}_res.ms"
    calibrated_data_image = f"{output_file_prefix}_calibrated"
    model_image = f"{output_file_prefix}_model"
    res_image = f"{output_file_prefix}_res"

    # Get antennas to flag based on Andrea's autocorrelation metrics
    if date is None:  # Attempt to parse the filename to get date
        date = datetime.datetime(
            int(output_file_prefix[:4]),
            int(output_file_prefix[4:6]),
            int(output_file_prefix[6:8]),
        )
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    bad_ant_list = get_bad_antenna_list(year, month, day)

    if run_aoflagger:
        os.system(f"aoflagger {datafile_path}")

    if output_dir is None:  # Use same directory as the data
        output_dir = os.path.dirname(datafile_path)
    if not os.path.isdir(output_dir):  # Make directory if it doesn't already exist
        os.mkdir(output_dir)

    if os.path.isdir(f"{output_dir}/{model_filename}"):
        print(f"Model file exists. Using {output_dir}/{model_filename}.")
    else:
        get_model_visibilities(
            model_visilibility_mode="run simulation",
            model_vis_file=f"{output_dir}/{model_filename}",
            include_diffuse=False,
            data_file=datafile_path,
            skymodel_path=skymodel_path,
            beam_path=beam_path,
        )

    # Read data
    uv = pyuvdata.UVData()
    print(f"Reading file {datafile_path}.")
    uv.read(f"{datafile_path}", data_column="DATA")
    uv.set_uvws_from_antenna_positions(update_vis=False)
    uv.data_array = np.conj(uv.data_array)  # Required to match pyuvdata conventions
    uv.phase_to_time(np.mean(uv.time_array))

    # Read model
    model_uv = pyuvdata.UVData()
    print(f"Reading file {output_dir}/{model_filename}.")
    model_uv.read(f"{output_dir}/{model_filename}", data_column="DATA")
    model_uv.set_uvws_from_antenna_positions(update_vis=False)
    model_uv.phase_to_time(np.mean(uv.time_array))

    # Flag antennas
    flag_antennas(
        uv,
        antenna_names=bad_ant_list,
        inplace=True,
    )

    # Calibrate
    uvcal = calibration_wrappers.sky_based_calibration_wrapper(
        uv,
        model_uv,
        min_cal_baseline_lambda=10,
        max_cal_baseline_lambda=125,
        gains_multiply_model=True,
        verbose=True,
        get_crosspol_phase=False,
        log_file_path=f"{output_dir}/{calibration_log}",
        xtol=1e-6,
        maxiter=200,
        antenna_flagging_iterations=0,
        parallel=False,
        lambda_val=1,
    )
    uvcal.write_calfits(calfits_filename, clobber=True)

    # Apply calibration
    pyuvdata.utils.uvcalibrate(uv, uvcal, inplace=True, time_check=False)
    if not os.path.isdir(calibrated_data_ms):
        uv.write_ms(calibrated_data_ms, fix_autos=True)
    # Image calibrated data
    os.system(
        f"/opt/bin/wsclean -pol I -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 -weight briggs 0 -no-update-model-required -mem 10 -no-reorder -name {calibrated_data_image} {calibrated_data_ms}"
    )
    os.system(
        f"/opt/bin/wsclean -pol I -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 -weight briggs 0 -no-update-model-required -mem 10 -no-reorder -name {model_image} {model_filename}"
    )

    # Subtract model from data
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
    if not os.path.isdir(res_ms):
        uv.write_ms(res_ms, fix_autos=True)
    os.system(
        f"/opt/bin/wsclean -pol I -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 -weight briggs 0 -no-update-model-required -mem 10 -no-reorder -name {res_image} {res_ms}"
    )


if __name__ == "__main__":
    calibration_pipeline(
        "/lustre/21cmpipe/2025-05-08/20250508_160736-160926_41MHz.ms",
        run_aoflagger=False,
    )
