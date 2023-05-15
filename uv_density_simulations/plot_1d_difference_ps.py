import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def plot_difference_ps():

    data_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims__cal_error_May2023_minus_May2023/data/1d_binning"
    uv_spacings = ["10", "5", "1", "0.5"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    names = uv_spacings
    for file_ind, spacing in enumerate(uv_spacings):
        filename = f"{data_path}/sim_uv_spacing_{spacing}_short_bls__calerrorgridded_minus_griddeduvfnoimgclip__dirty_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave"
        power = scipy.io.readsav(filename)["power"]
        k_edges = scipy.io.readsav(filename)["k_edges"]
        power_plot = np.repeat(power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(
            k_edges_plot, power_plot, color=colors[file_ind], label=names[file_ind]
        )
        plt.xscale("log")
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/cal_error_1d_ps_diff.png")


def plot_difference_ratio_ps():

    cal_error_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023"
    )
    reference_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"
    )
    uv_spacings = ["10", "5", "1", "0.5"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    names = uv_spacings
    for file_ind, spacing in enumerate(uv_spacings):
        cal_error_kpar0_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
        cal_error_ps_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda10-50_1dkpower.idlsave"
        reference_kpar0_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
        reference_ps_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda10-50_1dkpower.idlsave"

        cal_error_kpar0_power = scipy.io.readsav(cal_error_kpar0_path)["power"]
        k_edges = scipy.io.readsav(cal_error_kpar0_path)["k_edges"]
        cal_error_ps = scipy.io.readsav(cal_error_ps_path)["power"]
        k_edges_new = scipy.io.readsav(cal_error_ps_path)["k_edges"]
        if np.max(np.abs(k_edges_new - k_edges)) > 0:
            print("ERROR: k_edges mismatch!")
            print(f"Old k_edges: {k_edges}")
            print(f"New k_edges: {k_edges_new}")
        cal_error_fractional_power = cal_error_ps / cal_error_kpar0_power

        reference_kpar0_power = scipy.io.readsav(reference_kpar0_path)["power"]
        k_edges_new = scipy.io.readsav(reference_kpar0_path)["k_edges"]
        if np.max(np.abs(k_edges_new - k_edges)) > 0:
            print("ERROR: k_edges mismatch!")
            print(f"Old k_edges: {k_edges}")
            print(f"New k_edges: {k_edges_new}")
        reference_ps = scipy.io.readsav(reference_ps_path)["power"]
        k_edges_new = scipy.io.readsav(reference_ps_path)["k_edges"]
        if np.max(np.abs(k_edges_new - k_edges)) > 0:
            print("ERROR: k_edges mismatch!")
            print(f"Old k_edges: {k_edges}")
            print(f"New k_edges: {k_edges_new}")
        reference_fractional_power = reference_ps / reference_kpar0_power

        diff_ratio = cal_error_fractional_power - reference_fractional_power

        plot_vals = np.repeat(diff_ratio, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/cal_error_diff_ratio.png")


if __name__ == "__main__":

    plot_difference_ratio_ps()
