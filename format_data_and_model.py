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
    for filename in use_files:
        os.system(f"cp -r {datadir}/{filename} {copied_data_dir}/{filename}")
    use_files_full_paths = [f"{copied_data_dir}/{filename}" for filename in use_files]
    uvd = LWA_preprocessing.convert_raw_ms_to_uvdata(
        use_files_full_paths, data_column="DATA", combine_spws=True, run_aoflagger=True
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
