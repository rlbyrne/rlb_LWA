import pyuvdata
import numpy as np
import os
import sys

sys.path.append("LWA_data_preprocessing")
import LWA_preprocessing

subbands = ["41", "46", "50", "55", "59", "64", "69", "73", "78", "82"]
use_time_offsets = np.arange(0, 12)

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

for use_band in subbands:
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

    use_files = use_files[:1]  # added for debugging

    for filename in use_files:
        if not os.path.isfile(f"{copied_data_dir}/{filename}"):
            os.system(f"cp -r {datadir}/{filename} {copied_data_dir}/{filename}")
    use_files_full_paths = [f"{copied_data_dir}/{filename}" for filename in use_files]
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
        use_files_full_paths, data_column="DATA", combine_spws=True, #run_aoflagger=True
    )
    LWA_preprocessing.flag_antennas(
        uvd,
        antenna_names=flag_ants,
        flag_pol="all",
        inplace=True,
    )
    output_filename = f"20240303_{use_files[0].split('_')[-2]}-{use_files[-1].split('_')[-2]}_{use_band}MHz.uvfits"
    uvd.write_uvfits(
        f"{copied_data_dir}/{output_filename}",
        fix_autos=True,
    )
    if os.path.isfile(f"{copied_data_dir}/{output_filename}"):  # Delete copied files
        for filename in enumerate(use_files):
            os.system(f"rm -r {datadir}/{filename} {copied_data_dir}/{filename}")

    # Get model
    model_filepath = "/lustre/rbyrne/simulation_outputs"
    model_filenames = os.listdir(model_filepath)
    model_filenames = [filename for filename in model_filenames if filename.endswith(f"{use_band}MHz_source_sim.uvfits")]
    model_lsts = []
    model_lst_filenames = []
    for model_file in model_filenames:
        model_uv = pyuvdata.UVData()
        model_uv.read(f"{model_filepath}/{model_file}", read_data=False)
        lsts_new = list(set(model_uv.lst_array))
        model_lsts = np.concatenate((model_lsts, lsts_new))
        filenames_new = np.repeat([model_file], len(lsts_new))
        model_lst_filenames = np.concatenate((model_lst_filenames, filenames_new))

    model_uv_list = []
    for time_ind, use_lst in enumerate(list(set(uvd.lst_array))):
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
        model1_uv_diffuse.read(f"{model_filepath}/{model_filename1.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits")
        model1_uv_diffuse.select(lsts=[lst1])
        model1_uv_diffuse.filename = [""]
        model1_uv.sum_vis(model1_uv_diffuse, inplace=True)

        model2_uv = pyuvdata.UVData()
        model2_uv.read(f"{model_filepath}/{model_filename2}")
        model2_uv.select(lsts=[lst2])
        model2_uv.filename = [""]
        model2_uv_diffuse = pyuvdata.UVData()
        model2_uv_diffuse.read(f"{model_filepath}/{model_filename2.removesuffix('source_sim.uvfits')}diffuse_sim.uvfits")
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
    data_file_name = f"{copied_data_dir}/{output_filename.replace('.uvfits', '_model.uvfits')}"
    combined_model_uv.write_uvfits(data_file_name)
