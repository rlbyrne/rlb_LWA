import casatasks
import os
import sys

use_freq_bands = [
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

for freq_band in use_freq_bands:

    year = "2025"
    month = "05"
    day = "05"
    min_time_str = "123005"
    max_time_str = "123205"

    datadir = f"/lustre/pipeline/slow/{freq_band}MHz/{year}-{month}-{day}/12"
    copied_data_dir = f"/lustre/rbyrne/{year}-{month}-{day}"

    if not os.path.isdir(copied_data_dir):  # Make target directory if it does not exist
        os.mkdir(copied_data_dir)

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
        print("WARNING: Number of files found is not 12.")
        sys.exit()

    use_files.sort()
    output_filename = f"{year}{month}{day}_{use_files[0].split('_')[1]}-{use_files[-1].split('_')[1]}_{freq_band}MHz.ms"

    if not os.path.isdir(f"{copied_data_dir}/{output_filename}"):
        # Copy files
        for filename in use_files:
            if not os.path.isfile(f"{copied_data_dir}/{filename}"):
                print(f"Copying file {filename}")
                os.system(f"cp -r {datadir}/{filename} {copied_data_dir}/{filename}")
        use_files_full_paths = [f"{copied_data_dir}/{filename}" for filename in use_files]

        # Concatenate files
        casatasks.virtualconcat(
            vis=use_files_full_paths, concatvis=f"{copied_data_dir}/{output_filename}"
        )

        # Run aoflagger
        os.system(f"aoflagger {copied_data_dir}/{output_filename}")
