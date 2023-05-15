import numpy as np
import scipy.io
import matplotlib.pyplot as plt


data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims__cal_error_May2023_minus_May2023/data/1d_binning"
uv_spacings = ["10", "5", "1", "0.5"]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
names = uv_spacings
for spacing in uv_spacings:
    filename = f"{data_path}/sim_uv_spacing_{spacing}_short_bls__calerrorgridded_minus_griddeduvfnoimgclip__dirty_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave"
    power = scipy.io.readsav(filename)["power"]
    k_edges = scipy.io.readsav(filename)["k_edges"]
    power_plot = np.repeat(power, 2)
    k_edges_plot = np.concatenate(
        ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
    )
    plt.plot(k_edges_plot, power_plot, color=colors[file_ind], label=names[file_ind])
    plt.xscale("log")
plt.legend()
plt.savefig("/home/rbyrne/uv_density_sim_plots/cal_error_1d_ps_diff.png")
