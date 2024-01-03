import numpy as np
import scipy.io
import matplotlib.pyplot as plt

plot_save_dir = "/home/rbyrne/kpar0_plots_Dec2023"
run_filepaths = [
    "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_test_diffuse_Dec2023",
    "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_test_diffuse_Kelvin_conversion_Dec2023",
]
run_names = ["", "Kelvin conversion"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for run_ind, run_path in enumerate(run_filepaths):
    dirty_filename = f"{run_path}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
    residual_filename = f"{run_path}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_res_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
    dirty = scipy.io.readsav(dirty_filename)["power"]
    residual = scipy.io.readsav(residual_filename)["power"]
    k_edges = scipy.io.readsav(dirty_filename)["k_edges"]
    frac_power = (1.0 - np.sqrt(residual / dirty)) * 100.0
    power_plot = np.repeat(frac_power, 2)
    k_edges_plot = np.concatenate(
        ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
    )
    plt.plot(k_edges_plot, power_plot, color=colors[run_ind], label=run_names[run_ind])
    plt.xscale("log")
    plt.xlabel("k-perpendicular (h/Mpc)")
    plt.ylabel("Fractional Power Recovered (%)")
plt.legend()
plt.savefig(f"{plot_save_dir}/frac_power_recovered_Kelvin_conv_test.png")
plt.close()

for run_ind, run_path in enumerate(run_filepaths):

    residual_filename = f"{run_path}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_res_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
    residual = scipy.io.readsav(residual_filename)["power"]
    k_edges = scipy.io.readsav(residual_filename)["k_edges"]
    power_plot = np.repeat(residual, 2)
    k_edges_plot = np.concatenate(
        ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
    )
    plt.plot(
        k_edges_plot,
        power_plot,
        color=colors[run_ind],
        label=f"{run_names[run_ind]} residual",
    )

    dirty_filename = f"{run_path}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
    dirty = scipy.io.readsav(dirty_filename)["power"]
    k_edges = scipy.io.readsav(dirty_filename)["k_edges"]
    power_plot = np.repeat(dirty, 2)
    k_edges_plot = np.concatenate(
        ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
    )
    plt.plot(
        k_edges_plot,
        power_plot,
        "--",
        color=colors[run_ind],
        label=f"{run_names[run_ind]} dirty",
    )

    model_filename = f"{run_path}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_model_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
    model = scipy.io.readsav(model_filename)["power"]
    k_edges = scipy.io.readsav(model_filename)["k_edges"]
    power_plot = np.repeat(model, 2)
    k_edges_plot = np.concatenate(
        ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
    )
    plt.plot(
        k_edges_plot,
        power_plot,
        ":",
        color=colors[run_ind],
        label=f"{run_names[run_ind]} model",
    )

    plt.xscale("log")
    plt.xlabel("k-perpendicular (h/Mpc)")
    plt.ylabel("Power")
plt.legend()
plt.savefig(f"{plot_save_dir}/kpar0_power_Kelvin_conv_test.png")
plt.close()
