import numpy as np
import pyuvdata

use_freqs = [13, 18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
for freq in use_freqs:
    use_dir = f"/lustre/pipeline/cosmology/{freq}MHz/2026-04-19/11"
    filenames = [
        f"{use_dir}/20260419_112543_{freq}MHz.ms",
        f"{use_dir}/20260419_112553_{freq}MHz.ms",
        f"{use_dir}/20260419_112603_{freq}MHz.ms",
        f"{use_dir}/20260419_112613_{freq}MHz.ms",
        f"{use_dir}/20260419_112623_{freq}MHz.ms",
        f"{use_dir}/20260419_112633_{freq}MHz.ms",
    ]
    for file_ind, filename in enumerate(filenames):
        uv_new = pyuvdata.UVData()
        uv_new.read(filename)
        if file_ind == 0:
            uv = uv_new
        else:
            uv_new.phase_to_time(np.min(uv.time_array))
            uv.fast_concat(uv_new, "blt", inplace=True, run_check=False)

    uv.phase_to_time(np.mean(uv.time_array))
    uv.write_ms(f"/fast/rbyrne/20260419_112543-112633_{freq}MHz.ms")
