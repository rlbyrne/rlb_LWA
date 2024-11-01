import pyuvdata
import numpy as np
import os
import sys

sys.path.append("/home/rbyrne/rlb_LWA/LWA_data_preprocessing")
import LWA_preprocessing


def process_data_files(
    use_band,  # One of ["41", "46", "50", "55", "59", "64", "69", "73", "78", "82"]
):
    min_time_str = "093000"
    max_time_str = "093200"

    flag_ants = [
        "LWA009",
        "LWA041",
        "LWA044",
        "LWA052",
        "LWA058",
        "LWA076",
        "LWA095",
        "LWA105",
        "LWA111",
        "LWA120",
        "LWA124",
        "LWA138",
        "LWA150",
        "LWA159",
        "LWA191",
        "LWA204",
        "LWA208",
        "LWA209",
        "LWA232",
        "LWA234",
        "LWA255",
        "LWA267",
        "LWA280",
        "LWA288",
        "LWA292",
        "LWA302",
        "LWA307",
        "LWA309",
        "LWA310",
        "LWA314",
        "LWA325",
        "LWA341",
        "LWA352",
        "LWA364",
        "LWA365",
    ]

    datadir = f"/lustre/xhall/2024-03-02_rainy_day_data/{use_band}MHz/2024-03-03"
    copied_data_dir = "/lustre/rbyrne/2024-03-03"

    all_files = os.listdir(datadir)
    use_files = [
        filename
        for filename in all_files
        if filename.startswith("20240303") and filename.endswith(".ms")
    ]
    use_files = [
        filename
        for filename in use_files
        if (int(filename.split("_")[-2]) >= int(min_time_str))
        and int(filename.split("_")[-2]) < int(max_time_str)
    ]
    use_files.sort()
    output_filename = f"20240303_{use_files[0].split('_')[-2]}-{use_files[-1].split('_')[-2]}_{use_band}MHz.uvfits"

    if os.path.isfile(f"{copied_data_dir}/{output_filename}"):
        print(f"Reading data file {copied_data_dir}/{output_filename}.")
        uvd = pyuvdata.UVData()
        uvd.read(f"{copied_data_dir}/{output_filename}")
    else:  # Generate combined file
        for filename in use_files:
            if not os.path.isfile(f"{copied_data_dir}/{filename}"):
                print(f"Copying file {filename}.")
                os.system(f"cp -r {datadir}/{filename} {copied_data_dir}/{filename}")
        use_files_full_paths = [
            f"{copied_data_dir}/{filename}" for filename in use_files
        ]
        print(f"Generating combined data file.")
        uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
            use_files_full_paths,
            data_column="DATA",
            combine_spws=True,
            run_aoflagger=True,
            # conjugate_data=True,  # Add in for next run
        )
        LWA_preprocessing.flag_antennas(
            uvd,
            antenna_names=flag_ants,
            flag_pol="all",
            inplace=True,
        )
        print(f"Saving data file to {copied_data_dir}/{output_filename}.")
        uvd.write_uvfits(
            f"{copied_data_dir}/{output_filename}",
            fix_autos=True,
            force_phase=True,
        )
        if os.path.isfile(
            f"{copied_data_dir}/{output_filename}"
        ):  # Delete copied files
            for filename in use_files:
                os.system(f"rm -r {copied_data_dir}/{filename}")

    model_file_name = (
        f"{copied_data_dir}/{output_filename.replace('.uvfits', '_model.uvfits')}"
    )
    if not os.path.isfile(model_file_name):  # Get model
        model_filepath = "/lustre/rbyrne/simulation_outputs"
        lst_lookup_table_path = f"{model_filepath}/lst_lookup_table.csv"
        with open(lst_lookup_table_path, "r") as f:
            lst_data = f.readlines()
        model_lsts = np.array([])
        model_lst_filenames = np.array([])
        for line in lst_data[1:]:
            line_split = line.replace("\n", "").strip().split(",")
            model_lsts = np.append(model_lsts, float(line_split[0]))
            model_lst_filenames = np.append(
                model_lst_filenames,
                f"{line_split[1]}_{use_band}MHz_source_sim.uvfits".strip(),
            )

        model_uv_list = []
        for time_ind, use_lst in enumerate(list(set(uvd.lst_array))):
            print(
                f"Calculating model visibilities for time step {time_ind+1} of {len(list(set(uvd.lst_array)))}."
            )
            lst_distance = np.abs(model_lsts - use_lst)
            ind1 = np.where(lst_distance == np.min(lst_distance))[0]
            ind2 = np.where(lst_distance == np.sort(lst_distance)[1])[0]
            lst1 = model_lsts[ind1]
            model_filename1 = model_lst_filenames[ind1][0]
            lst2 = model_lsts[ind2]
            model_filename2 = model_lst_filenames[ind2][0]

            # Interpolate models
            lst_spacing = np.abs(lst2 - lst1)
            lst_spacing1 = np.abs(lst1 - use_lst)
            lst_spacing2 = np.abs(lst2 - use_lst)
            model1_uv = pyuvdata.UVData()
            model1_uv.read(f"{model_filepath}/{model_filename1}")
            model1_uv.select(lsts=[lst1])
            model1_uv.filename = [""]
            model1_uv_diffuse = pyuvdata.UVData()
            model1_uv_diffuse.read(
                f"{model_filepath}/{model_filename1.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits"
            )
            model1_uv_diffuse.select(lsts=[lst1])
            model1_uv_diffuse.filename = [""]
            model1_uv.sum_vis(model1_uv_diffuse, inplace=True)

            model2_uv = pyuvdata.UVData()
            model2_uv.read(f"{model_filepath}/{model_filename2}")
            model2_uv.select(lsts=[lst2])
            model2_uv.filename = [""]
            model2_uv_diffuse = pyuvdata.UVData()
            model2_uv_diffuse.read(
                f"{model_filepath}/{model_filename2.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits"
            )
            model2_uv_diffuse.select(lsts=[lst2])
            model2_uv_diffuse.filename = [""]
            model2_uv.sum_vis(model2_uv_diffuse, inplace=True)

            # Phase to consistent phase center
            phase_center_time = np.mean(uvd.time_array)
            model1_uv.phase_to_time(phase_center_time)
            model2_uv.phase_to_time(phase_center_time)

            # Combine data
            model1_uv.data_array *= lst_spacing1 / lst_spacing
            model2_uv.data_array *= lst_spacing2 / lst_spacing
            model_uv = model1_uv.sum_vis(
                model2_uv,
                inplace=False,
                run_check=False,
                check_extra=False,
                run_check_acceptability=False,
                override_params=["lst_array", "time_array", "uvw_array", "filename"],
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
                override_params=["lst_array", "time_array", "uvw_array", "filename"],
            )
            model_uv.data_array *= np.abs(model_uv_abs.data_array) / np.abs(
                model_uv.data_array
            )

            model_uv.time_array = np.full(
                model_uv.Nblts, np.sort(list(set(uvd.time_array)))[time_ind]
            )
            model_uv.lst_array = np.full(model_uv.Nblts, use_lst)
            model_uv_list.append(model_uv)

        # Save output
        combined_model_uv = model_uv_list[0]
        if len(model_uv_list) > 1:
            for model_uv_use in model_uv_list[1:]:
                combined_model_uv.fast_concat(model_uv_use, "blt", inplace=True)

        print(f"Saving model file to {model_file_name}.")
        combined_model_uv.write_uvfits(model_file_name, force_phase=True)


def create_model_lst_lookup_table():

    model_filepath = "/lustre/rbyrne/simulation_outputs"
    use_band = "41"
    model_filenames = os.listdir(model_filepath)
    model_filenames = [
        filename
        for filename in model_filenames
        if filename.endswith(f"{use_band}MHz_source_sim.uvfits")
    ]
    output_filename = f"{model_filepath}/lst_lookup_table.csv"
    with open(output_filename, "w") as f:
        f.write("LST, filename \n")
    for model_file in model_filenames:
        model_uv = pyuvdata.UVData()
        model_uv.read(f"{model_filepath}/{model_file}", read_data=False)
        lsts_new = list(set(model_uv.lst_array))
        for lst in lsts_new:
            with open(output_filename, "a") as f:
                f.write(
                    f"{lst}, {model_file.removesuffix(f'_{use_band}MHz_source_sim.uvfits')} \n"
                )


def convert_to_ms():
    uv = pyuvdata.UVData()
    # uv.read("/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz_model.uvfits")
    # uv.reorder_pols(order="CASA")
    # uv.write_ms("/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz_model.ms")
    # uv = pyuvdata.UVData()
    # uv.read("/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz.uvfits")
    # uv.reorder_pols(order="CASA")
    # uv.write_ms("/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz.ms")
    uv.read("/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz_calibrated.uvfits")
    uv.reorder_pols(order="CASA")
    uv.write_ms("/lustre/rbyrne/2024-03-03/20240303_093000-093151_41MHz_calibrated.ms")


def combine_subbands():

    freq_bands = [
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
    for freq_band_ind, use_freq_band in enumerate(freq_bands):
        uvd_new = pyuvdata.UVData()
        uvd_new.read(
            f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{use_freq_band}MHz_calibrated.uvfits"
        )
        LWA_preprocessing.flag_outriggers(uvd_new, remove_outriggers=True, inplace=True)
        if freq_band_ind == 0:
            uvd = uvd_new
        else:
            uvd.fast_concat(uvd_new, "freq", inplace=True)
    uvd.write_uvfits(
        f"/lustre/rbyrne/2024-03-03/20240303_093000-093151_{freq_bands[0]}-{freq_bands[-1]}MHz_calibrated_core.uvfits",
        fix_autos=True,
    )


if __name__ == "__main__":
    #args = sys.argv
    #use_band = args[1]
    #process_data_files(use_band)
    combine_subbands()