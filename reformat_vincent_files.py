import pyuvdata

obsids_list = [
    "density1.5_gleam",
    "density1.5_gsm08",
    "density1.75_gleam",
    "density1.75_gsm08",
    "density2.0_gleam",
    "density2.0_gsm08",
    "density3.0_gleam",
    "density3.0_gsm08",
    "hexa_gleam",
    "hexa_gsm08",
    "pos_error1e-3_gleam",
    "pos_error1e-3_gsm08",
    "random_gleam",
    "random_gsm08",
]

for obs in obsids_list:
    uv = pyuvdata.UVData()
    uv.read(f"/lustre/rbyrne/vincent_sims/{obs}.uvh5")
    uv.write_uvfits(f"/lustre/rbyrne/vincent_sims/{obs}.uvfits", force_phase=True)