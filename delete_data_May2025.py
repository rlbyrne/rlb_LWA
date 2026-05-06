import numpy as np
import os

dates_to_delete = [
    "2024-04-18",
    "2024-04-19",
    "2024-04-20",
    "2024-04-23",
    "2024-04-24",
    "2024-04-25",
    "2024-04-26",
    "2024-04-27",
    "2024-04-29",
    "2024-04-30",
]

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

for date in dates_to_delete:
    for freq in use_freq_bands:
        filename = f"/lustre/pipeline/cosmology/{freq}MHz/{date}"
        if os.path.isdir(filename):
            print(f"Deleting {filename}")
            os.system(f"sudo rm -r {filename}")
