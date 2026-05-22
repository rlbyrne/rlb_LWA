import numpy as np
import os

dates_to_delete = [
    "2024-05-01",
    "2024-05-05",
    "2024-05-06",
    "2024-05-07",
    "2024-05-08",
    "2024-05-12",
    "2024-05-13",
    "2024-05-14",
    "2024-05-15",
    "2024-05-17",
    "2024-05-18",
    "2024-05-20",
    "2024-05-21",
    "2024-05-24",
    "2024-05-25",
    "2024-05-26",
    "2024-05-28",
    "2024-05-29",
    "2024-06-01",
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
