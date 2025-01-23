import casatasks
import os

use_freq_bands = [
    #"41",
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

for freq_band in use_freq_bands:

    min_time_str = "093000"
    max_time_str = "093200"

    datadir = f"/lustre/xhall/2024-03-02_rainy_day_data/{freq_band}MHz/2024-03-03"
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
    output_filename = f"20240303_{use_files[0].split('_')[-2]}-{use_files[-1].split('_')[-2]}_{freq_band}MHz.ms"

    if not os.path.isdir(f"{copied_data_dir}/{output_filename}"):
        # Copy files
        for filename in use_files:
            if not os.path.isfile(f"{copied_data_dir}/{filename}"):
                print(f"Copying file {filename}.")
                os.system(f"cp -r {datadir}/{filename} {copied_data_dir}/{filename}")
        use_files_full_paths = [f"{copied_data_dir}/{filename}" for filename in use_files]

        # Concatenate files
        casatasks.virtualconcat(
            vis=use_files_full_paths, concatvis=f"{copied_data_dir}/{output_filename}"
        )

        # Run aoflagger
        os.system(f"aoflagger {copied_data_dir}/{output_filename}")
