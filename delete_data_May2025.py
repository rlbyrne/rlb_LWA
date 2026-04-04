import numpy as np
import os

dates_to_delete = [
    "2024-04-10",
    "2024-04-11",
    "2024-04-12",
    "2024-04-13",
    "2024-04-15",
    "2024-04-16",
    "2024-04-17",
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
