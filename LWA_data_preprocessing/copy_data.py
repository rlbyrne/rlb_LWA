import numpy as np
import os

data_filenames = os.listdir("/data06/slow")
data_filenames = np.sort(data_filenames)
data_filenames = [file for file in data_filenames if "73MHz" in file]
for file in data_filenames:
    os.system(
        f"scp -r /data06/slow/{file} 10.41.0.85:/data10/rbyrne/2023-08-11_24hour_run/"
    )
new_filenames = os.listdir("/data06/slow")
new_filenames = [file for file in new_filenames if "73MHz" in file]
new_filenames = [file for file in new_filenames if file not in data_filenames]
for file in new_filenames:
    os.system(
        f"scp -r /data06/slow/{file} 10.41.0.85:/data10/rbyrne/2023-08-11_24hour_run/"
    )
