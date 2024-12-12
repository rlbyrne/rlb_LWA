import numpy as np
import os


file_directory = "/lustre/rbyrne/simulation_outputs/to_rename"
output_directory = "/lustre/rbyrne/simulation_outputs"
filenames = os.listdir(file_directory)
filenames = [use_file for use_file in filenames if use_file.endswith(".uvfits")]
output_filenames = []
for use_file in filenames:
    timestamp = use_file.split("_")[1]
    delta_time = int(timestamp) - 93000
    delta_time_seconds = delta_time % 60
    delta_time_minutes = ((delta_time - delta_time_seconds) / 60) % 60
    delta_time_hours = (
        (delta_time - delta_time_minutes * 60 - delta_time_seconds) / 60**2
    ) % 24

    # Reference time is 09:30:00
    time_hours = 9 + delta_time_hours
    time_minutes = 30 + delta_time_minutes
    time_seconds = 0 + delta_time_seconds
    if time_seconds > 59:
        time_minutes += int(np.floor(time_seconds / 60.0))
        time_seconds = time_seconds % 60
    if time_minutes > 59:
        time_hours += int(np.floor(time_minutes / 60.0))
        time_minutes = time_minutes % 60
    time_hours = time_hours % 24
    time_string = f"{str(int(time_hours)).zfill(2)}{str(int(time_minutes)).zfill(2)}{str(int(time_seconds)).zfill(2)}"

    output_filename = f"{use_file[0:9]}{time_string}{use_file[15:]}"
    os.system(f"mv {file_directory}/{use_file} {output_directory}/{output_filename}")
