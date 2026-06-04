import numpy as np
import os
import sys

use_freq_bands = [
    "13",
    "18",
    "23",
    "27",
    "32",
    "36",
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

dates_to_delete = [
    "2026-04-22",
    "2026-05-29",
    "2026-05-30",
    "2026-05-31",
    "2026-06-01",
]

for date in dates_to_delete:
    for freq in use_freq_bands:
        filename = f"/lustre/pipeline/cosmology/{freq}MHz/{date}"
        if os.path.isdir(filename):
            print(f"Deleting {filename}")
            os.system(f"sudo rm -r {filename}")
