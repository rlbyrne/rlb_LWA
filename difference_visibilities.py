import pyuvdata
import numpy as np

data_file = "/safepool/rbyrne/pyuvsim_sims_Dec2023/20230819_093023_73MHz_calibrated.uvfits"
data = pyuvdata.UVData()
data.read(data_file)

data.reorder_pols()
data.reorder_blts()
data.reorder_freqs(channel_order="freq")
data.nsample_array[:, :, :, :] = 1.0

#model_names = ["cyg_cas", "deGasperin_cyg_cas", "deGasperin_sources", "VLSS"]
#model_files = [f"/safepool/rbyrne/pyuvsim_sims_Dec2023/20230819_093023_73MHz_{name}_sim.uvfits" for name in model_names]
model_names = ["deGasperin_cyg_cas_NMbeam"]
model_files = ["/safepool/rbyrne/pyuvsim_sims_Dec2023/20230819_093023_73MHz_deGasperin_cyg_cas_sim_NMbeam.uvfits"]
for model_ind, model_file in enumerate(model_files):
    model = pyuvdata.UVData()
    model.read(model_file)
    model.reorder_pols()
    model.reorder_blts()
    model.reorder_freqs(channel_order="freq")
    model.flag_array = data.flag_array
    model.vis_units = data.vis_units
    model.filename = data.filename
    diff = data.sum_vis(model, difference=True, inplace=False)
    diff.write_uvfits(f"/safepool/rbyrne/pyuvsim_sims_Dec2023/20230819_093023_73MHz_data_minus_{model_names[model_ind]}.uvfits")
