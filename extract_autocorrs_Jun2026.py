import pyuvdata
import os
import pathlib

date = "2026-04-19"
hours = ["05", "06", "07", "08", "11"]
use_freqs = [
    "34",
    "44",
    "52",
    "62",
    "72",
    "79",
    "83",
]
for freq in use_freqs:
    data_dir = pathlib.Path(f"/lustre/pipeline/cosmology/concatenated_data/{freq}MHz/{date}")
    data_paths = [str(p) for p in data_dir.rglob("*.ms") if p.is_dir()]
    if len(data_paths) > 0:
        uv = None
        for file_ind, path in enumerate(data_paths):
            print(f"Reading file {file_ind+1} of {len(data_paths)}")
            uv_new = pyuvdata.UVData()
            uv_new.read(path)
            uv_new.select(ant_str="auto")
            if file_ind == 0:
                uv = uv_new
            else:
                uv.fast_concat(uv_new, "blt", inplace=True)
        uv.write_ms(f"/lustre/rbyrne/2026-04-19/20260419_{freq}MHz_autocorrs.ms")


