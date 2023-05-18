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


def calculate_fractional_power(kpar0_power, k_edges_kpar0, ps, k_edges_ps):

    diff_vals = np.full((len(ps)), np.nan, dtype=float)
    for ind in range(len(ps)):
        k_bin_center = np.mean([k_edges_ps[ind], k_edges_ps[ind + 1]])
        kpar0_ind = np.where(
            (k_edges_kpar0[:-1] < k_bin_center) & (k_edges_kpar0[1:] > k_bin_center)
        )[0]
        if len(kpar0_ind) > 0:
            if kpar0_power[kpar0_ind] != 0:
                diff_vals[ind] = ps[ind] / kpar0_power[kpar0_ind]
    return diff_vals


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
        reference_kpar0_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
        reference_ps_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda10-50_1dkpower.idlsave"

        cal_error_kpar0_power = scipy.io.readsav(cal_error_kpar0_path)["power"]
        k_edges_kpar0 = scipy.io.readsav(cal_error_kpar0_path)["k_edges"]
        cal_error_ps = scipy.io.readsav(cal_error_ps_path)["power"]
        k_edges_ps = scipy.io.readsav(cal_error_ps_path)["k_edges"]
        cal_error_fractional_power = calculate_fractional_power(
            cal_error_kpar0_power, k_edges_kpar0, cal_error_ps, k_edges_ps
        )

        reference_kpar0_power = scipy.io.readsav(reference_kpar0_path)["power"]
        k_edges_kpar0 = scipy.io.readsav(reference_kpar0_path)["k_edges"]
        reference_ps = scipy.io.readsav(reference_ps_path)["power"]
        k_edges_ps_new = scipy.io.readsav(reference_ps_path)["k_edges"]
        if np.max(np.abs(k_edges_ps_new - k_edges_ps)) > 0:
            print("ERROR: k_edges mismatch!")
            print(f"Old k_edges: {k_edges_ps}")
            print(f"New k_edges: {k_edges_ps_new}")
        reference_fractional_power = calculate_fractional_power(
            reference_kpar0_power, k_edges_kpar0, reference_ps, k_edges_ps
        )

        diff_ratio = cal_error_fractional_power - reference_fractional_power

        plot_vals = np.repeat(diff_ratio, 2)
        k_edges_plot = np.concatenate(
            ([k_edges_ps[0]], np.repeat(k_edges_ps[1:-1], 2), [k_edges_ps[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/cal_error_diff_ratio.png")


def plot_ratio_ps():

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
        reference_kpar0_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
        reference_ps_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda10-50_1dkpower.idlsave"

        cal_error_kpar0_power = scipy.io.readsav(cal_error_kpar0_path)["power"]
        k_edges_kpar0 = scipy.io.readsav(cal_error_kpar0_path)["k_edges"]
        cal_error_ps = scipy.io.readsav(cal_error_ps_path)["power"]
        k_edges_ps = scipy.io.readsav(cal_error_ps_path)["k_edges"]
        cal_error_fractional_power = calculate_fractional_power(
            cal_error_kpar0_power, k_edges_kpar0, cal_error_ps, k_edges_ps
        )

        reference_kpar0_power = scipy.io.readsav(reference_kpar0_path)["power"]
        k_edges_kpar0 = scipy.io.readsav(reference_kpar0_path)["k_edges"]
        reference_ps = scipy.io.readsav(reference_ps_path)["power"]
        k_edges_ps_new = scipy.io.readsav(reference_ps_path)["k_edges"]
        if np.max(np.abs(k_edges_ps_new - k_edges_ps)) > 0:
            print("ERROR: k_edges mismatch!")
            print(f"Old k_edges: {k_edges_ps}")
            print(f"New k_edges: {k_edges_ps_new}")
        reference_fractional_power = calculate_fractional_power(
            reference_kpar0_power, k_edges_kpar0, reference_ps, k_edges_ps
        )

        diff_ratio = cal_error_fractional_power - reference_fractional_power

        plot_vals = np.repeat(reference_fractional_power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges_ps[0]], np.repeat(k_edges_ps[1:-1], 2), [k_edges_ps[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind], linestyle="dashed")
        plot_vals = np.repeat(cal_error_fractional_power, 2)
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/cal_error_ratio.png")


def plot_ps():

    cal_error_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023"
    )
    reference_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"
    )
    uv_spacings = ["10", "5", "1", "0.5"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    names = uv_spacings

    # Plot error-free version
    for file_ind, spacing in enumerate(uv_spacings):
        data_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda1-45_1dkpower.idlsave"

        k_edges = scipy.io.readsav(data_path)["k_edges"]
        power = scipy.io.readsav(data_path)["power"]

        plot_vals = np.repeat(power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim([1e4, 1e15])
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/ps_no_cal_error.png")
    plt.close()

    # Plot with cal error
    for file_ind, spacing in enumerate(uv_spacings):
        #data_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda10-50_1dkpower.idlsave"
        data_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda1-45_1dkpower.idlsave"

        k_edges = scipy.io.readsav(data_path)["k_edges"]
        power = scipy.io.readsav(data_path)["power"]

        plot_vals = np.repeat(power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim([1e4, 1e15])
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/ps_cal_error.png")
    plt.close()


def plot_kpar0():

    cal_error_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023"
    )
    reference_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"
    )
    uv_spacings = ["10", "5", "1", "0.5"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    names = uv_spacings

    # Plot error-free version
    for file_ind, spacing in enumerate(uv_spacings):
        data_path = f"{reference_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"

        k_edges = scipy.io.readsav(data_path)["k_edges"]
        power = scipy.io.readsav(data_path)["power"]

        plot_vals = np.repeat(power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim([1e10, 1e14])
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/kpar0_no_cal_error.png")
    plt.close()

    # Plot with cal error
    for file_ind, spacing in enumerate(uv_spacings):
        data_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"

        k_edges = scipy.io.readsav(data_path)["k_edges"]
        power = scipy.io.readsav(data_path)["power"]

        plot_vals = np.repeat(power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim([1e10, 1e14])
    plt.legend()
    plt.savefig("/home/rbyrne/uv_density_sim_plots/kpar0_cal_error.png")
    plt.close()


def plot_ps_modified_kernel():

    cal_error_path = (
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023"
    )
    uv_spacings = ["10", "5", "1", "0.5"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    names = uv_spacings

    # Plot with cal error
    for file_ind, spacing in enumerate(uv_spacings):
        #data_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda10-50_1dkpower.idlsave"
        data_path = f"{cal_error_path}/ps/data/1d_binning/sim_uv_spacing_{spacing}_short_bls_cal_error__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_no_horizon_wedge_kperplambda1-45_1dkpower.idlsave"

        k_edges = scipy.io.readsav(data_path)["k_edges"]
        power = scipy.io.readsav(data_path)["power"]

        plot_vals = np.repeat(power, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(k_edges_plot, plot_vals, color=colors[file_ind], label=names[file_ind])
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim([1e4, 1e15])
    plt.legend()
    plt.xlabel("k-perpendicular (h Mpc$^{-1}$)")
    plt.ylabel("P$_k$ mK$^2$ h$^{-3}$ Mpc$^3$")
    plt.savefig("/home/rbyrne/uv_density_sim_plots/ps_cal_error_modified_kernel.png")
    plt.close()


if __name__ == "__main__":

    plot_ps_modified_kernel()
