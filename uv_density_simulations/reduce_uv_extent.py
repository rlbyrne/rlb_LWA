import pyuvdata
import numpy as np


uv_spacings = ["0.5"]
max_bl_wl = 50.0
for spacing in uv_spacings:
    filename = (
        f"/safepool/rbyrne/uv_density_simulations/sim_uv_spacing_{spacing}.uvfits"
    )
    uv = pyuvdata.UVData()
    uv.read_uvfits(filename)
    freq = np.mean(uv.freq_array)
    wavelength = 3e8 / freq
    bl_lengths_wl = np.sqrt(np.sum(np.abs(uv.uvw_array) ** 2, axis=1)) / wavelength
    keep_bl = np.where(bl_lengths_wl <= max_bl_wl)[0]
    uv.select(blt_inds=keep_bl, inplace=True)
    print(f"Reducing total number of antennas from {uv.Nants_telescope} to {uv.Nants_data}.)
    uv.Nants_telescope = uv.Nants_data
    uv.antenna_numbers = np.arange(uv.Nants_data)
    uv.antenna_names = np.array([str(ind) for ind in np.arange(uv.Nants_data)])
    uv.check()
    uv.write_uvfits(
        f"/safepool/rbyrne/uv_density_simulations/sim_uv_spacing_{spacing}_short_bls.uvfits"
    )
