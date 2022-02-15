import pyuvdata
import os
import numpy as np

data_path = "/lustre/rbyrne/LWA_data_02102022"
freq = "70MHz"
start_time_stamp = 191447
end_time_stamp = 194824

filenames = os.listdir(data_path)
use_files = []
for file in filenames:
    file_split = file.split("_")
    if file_split[2] == "{}.ms.tar".format(freq):
        if int(file_split[1]) >= start_time_stamp and int(file_split[1]) <= end_time_stamp:
            use_files.append(file)

print(use_files)
