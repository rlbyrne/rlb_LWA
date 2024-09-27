import pyradiosky
import numpy as np

for freq_band in [
    "18",
    "23",
    "27",
    "36",
    "41",
    "46",
    "50",
    "55",
    "59",
    "64",
    "73",
    "78",
    "82",
]:
    use_map = pyradiosky.SkyModel()
    use_map.read(f"/lustre/rbyrne/skymodels/Gasperin2020_point_sources_plus_{freq_band}.skyh5")
    comp_ind = np.where([name.startswith("Cyg") for name in use_map.name])
    flux = use_map.stokes[0, 0, comp_ind][0][0]
    print(f"{use_map.name[comp_ind][0][:3]} flux, {freq_band} MHz: {flux}")