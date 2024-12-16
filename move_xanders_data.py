import os

directories = [
#    "/lustre/xhall/2023-08-19_24hour_run",  # removed
#    "/lustre/xhall/2023-11-22_48hour_run",  # removed
#    "/lustre/xhall/2023-12-21_solstice_data",  # some duplicate data
    "/lustre/xhall/2024-03-01_night_data",
]

all_files = []
for directory in directories:
    files = os.listdir(directory)
    ms_files = [f"{directory}/{use_file}" for use_file in files if use_file.endswith(".ms")]
    all_files.extend(ms_files)
    subdirs = [f"{directory}/{use_file}" for use_file in files if not use_file.endswith(".ms")]
    for subdir in subdirs:
        files = os.listdir(subdir)
        ms_files = [f"{subdir}/{use_file}" for use_file in files if use_file.endswith(".ms")]
        all_files.extend(ms_files)
        subsubdirs = [f"{subdir}/{use_file}" for use_file in files if not use_file.endswith(".ms")]
        for subsubdir in subsubdirs:
            files = os.listdir(subsubdir)
            ms_files = [f"{subsubdir}/{use_file}" for use_file in files if use_file.endswith(".ms")]
            all_files.extend(ms_files)

for use_file in all_files:
    use_file_name = use_file.split("/")[-1]
    year = use_file_name[:4]
    month = use_file_name[4:6]
    day = use_file_name[6:8]
    hour = use_file_name[9:11]
    subband = use_file_name[16:21]
    if not os.path.isdir(f"/lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/"):
        if not os.path.isdir(f"/lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/"):
            os.system(f"sudo mkdir /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/")
        os.system(f"sudo mkdir /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/")
    if os.path.isdir(f"/lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/{use_file_name}"):
        print(f"WARNING: File /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/{use_file_name} exists. Skipping.")
    else:
        print(
            f"Running command: sudo mv -n {use_file} /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/"
        )
        os.system(
            f"sudo mv -n {use_file} /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/"
        )
