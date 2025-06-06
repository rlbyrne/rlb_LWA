import os
import subprocess
import ast
import sys
import numpy as np
import pyuvdata
from generate_model_vis_fftvis import run_fftvis_sim


def concatenate_and_flag_files(use_files_full_paths, output_filename):
    os.system(
        f"python concatenate_ms_files.py {use_files_full_paths} {output_filename}"
    )


def get_bad_antenna_list(
    year,
    month,
    day,
    conda_env="deployment",
    script_path="/opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/get_bad_ants.py",
):

    result = subprocess.getoutput(
        f"conda run -n {conda_env} python {script_path} {year} {month} {day}"
    )
    result = result.split("\n")
    result = [line for line in result if line.startswith("get_bandants output:")][0]
    result = result.split("get_bandants output:")[1].strip()
    result = ast.literal_eval(result)
    if np.isfinite(result[0]):
        return result[1]
    else:
        print(
            "ERROR: No information on bad antennas found for {year}/{month}/{day}. Exiting."
        )
        sys.exit(1)


def get_model_visibilities(
    model_visilibility_mode=None,
    model_file=None,
    output_model_filepath=None,
    include_diffuse=False,
    lst_lookup_table="/lustre/rbyrne/simulation_outputs/lst_lookup_table.csv",
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
    model_file : str
        Required and used only if model_visilibility_mode is "read file".
    output_model_filepath : str
        Required and used only if model_visilibility_mode is "LST interpolate"
        or "run simulation". Path to an output ms file. Model visibilities will
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
        model = pyuvdata.UVData()
        model.read(model_file)

    elif model_visilibility_mode == "LST interpolate":
        if os.path.isdir(output_model_filepath):
            print(f"ERROR: File {output_model_filepath} exits. Exiting.")
            sys.exit(1)
        if not os.path.isdir(os.path.dirname(output_model_filepath)):
            print(
                f"ERROR: Directory {os.path.dirname(output_model_filepath)} not found. Exiting."
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

        print(f"Saving model file to {output_model_filepath}.")
        combined_model_uv.write_ms(output_model_filepath, force_phase=True)

    elif model_visilibility_mode == "run simulation":
        if include_diffuse:
            compact_source_output_filepath = (
                f"{output_model_filepath.removesuffix('.ms')}compact.ms"
            )
            diffuse_output_filepath = (
                f"{output_model_filepath.removesuffix('.ms')}diffuse.ms"
            )
        else:
            compact_source_output_filepath = output_model_filepath
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
            compact_sim.write_ms(output_model_filepath, clobber=True, fix_autos=True)
            if os.path.isdir(output_model_filepath):  # Confirm write was successful
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
