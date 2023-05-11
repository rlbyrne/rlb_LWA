import numpy as np
import scipy.io
import matplotlib.pyplot as plt


reference_run_path = (
    "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"
)
unnormalized_run_path = (
    "/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_May2023"
)
normalized_run_path = (
    "/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_normalized_May2023"
)

obsid = "sim_uv_spacing_10_short_bls"
pol = "xx"

reference_kpar0_path = f"{reference_run_path}/ps/data/1d_binning/{obsid}__gridded_uvf_noimgclip_dirty_{pol}_dft_averemove_swbh_dencorr_k0power.idlsave"
unnormalized_kpar0_path = f"{unnormalized_run_path}/ps/data/1d_binning/{obsid}__gridded_uvf_noimgclip_dirty_{pol}_dft_averemove_swbh_dencorr_k0power.idlsave"
normalized_kpar0_path = f"{normalized_run_path}/ps/data/1d_binning/{obsid}__gridded_uvf_noimgclip_dirty_{pol}_dft_averemove_swbh_dencorr_k0power.idlsave"

colors = ["tab:blue", "tab:orange", "tab:green"]
names = ["reference", "beam error", "beam error, normalized"]
for file_ind, file_path in enumerate(
    [reference_kpar0_path, unnormalized_kpar0_path, normalized_kpar0_path]
):
    power = scipy.io.readsav(file_path)["power"]
    k_edges = scipy.io.readsav(file_path)["k_edges"]
    power_plot = np.repeat(power, 2)
    k_edges_plot = np.concatenate(
        ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
    )
    plt.plot(k_edges_plot, power_plot, color=colors[file_ind], label=names[file_ind])
    plt.xscale("log")
plt.legend()
plt.savefig("/home/rbyrne/kpar0_plot.png")
