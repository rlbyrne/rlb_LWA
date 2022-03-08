import pyuvdata
import numpy as np

path = "/safepool/rbyrne/LWA_pyuvsim_simulations/"
pyuvsim_output_file = "OVRO-LWA_1h_sim_results.uvfits"

uv = pyuvdata.UVData()
uv.read_uvfits(f"{path}/{pyuvsim_output_file}")
uv.unphase_to_drift()

times = sorted(list(set(uv.time_array)))
center_time = times[int(np.floor(len(times)/2))]

uv_center_time = uv.copy()
uv_center_time.select(times=center_time)
uv_center_time.phase_to_time(center_time)
uv_center_time.write_uvfits(f"{path}/OVRO-LWA_1h_sim_center_time.uvfits")

uv_unphased = uv.copy()
uv_unphased.downsample_in_time(n_times_to_avg=uv.Ntimes, allow_drift=True)
uv_unphased.phase_to_time(center_time)
uv_unphased.write_uvfits(f"{path}/OVRO-LWA_1h_sim_unphased.uvfits")

uv_phased = uv.copy()
uv_phased.phase_to_time(center_time)
uv_phased.downsample_in_time(n_times_to_avg=uv.Ntimes)
uv_phased.write_uvfits(f"{path}/OVRO-LWA_1h_sim_phased.uvfits")
