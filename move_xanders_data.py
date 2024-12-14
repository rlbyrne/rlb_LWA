import os

directories = [
    "/lustre/xhall/2023-08-19_24hour_run",
    "/lustre/xhall/2023-11-22_48hour_run",
    "/lustre/xhall/2023-12-21_solstice_data",
    "/lustre/xhall/2024-03-01_night_data",
]
dryrun = True

for directory in directories:
    files = os.listdir(directory)
    for use_file in files:
        year = use_file[:4]
        month = use_file[4:6]
        day = use_file[6:8]
        hour = use_file[9:11]
        subband = use_file[16:21]
        if dryrun:
            print(
                "sudo mv {directory}/{use_file} /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/"
            )
        else:
            os.system(
                "sudo mv {directory}/{use_file} /lustre/pipeline/night-time/{subband}/{year}-{month}-{day}/{hour}/"
            )
        break
