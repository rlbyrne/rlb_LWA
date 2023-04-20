import pyuvdata
import numpy as np


uv_spacings = ["0.5"]
max_bl_wl = 80.
for spacing in uv_spacings:
    filename = f"/safepool/rbyrne/uv_density_simulations/sim_uv_spacing_{spacing}.uvfits"
    uv = pyuvdata.UVData()
    uv.read_uvfits(filename)
    freq = np.mean(uv.freq_array)
    wavelength = 3e8 / freq
    bl_lengths_wl = np.sqrt(np.sum(np.abs(uv.uvw_array)**2, axis=1)) / wavelength
    keep_bl = np.where(bl_lengths_wl <= max_bl_wl)[0]
    uv.select(blt_inds=keep_bl, inplace=True)
    uv.write_uvfits(f"/safepool/rbyrne/uv_density_simulations/sim_uv_spacing_{spacing}_short_bls.uvfits")
