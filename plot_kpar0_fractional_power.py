import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def plot_kpar0_Jan2():

    plot_save_dir = "/home/rbyrne/kpar0_plots_Dec2023"
    run_filepaths = [
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_test_diffuse_debug_Kelvin_conversion_Jan2024",
        "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_test_diffuse_Kelvin_conversion_Dec2023",
    ]
    run_names = ["debug version", "main version"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    plot_id = "debug_test"

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
        plt.plot(
            k_edges_plot, power_plot, color=colors[run_ind], label=run_names[run_ind]
        )
        plt.xscale("log")
        plt.xlabel("k-perpendicular (h/Mpc)")
        plt.ylabel("Fractional Power Recovered (%)")
    plt.legend()
    plt.savefig(f"{plot_save_dir}/frac_power_recovered_{plot_id}.png")
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
        plt.yscale("log")
        plt.xlabel("k-perpendicular (h/Mpc)")
        plt.ylabel("Power")
    plt.legend()
    plt.savefig(f"{plot_save_dir}/kpar0_power_{plot_id}.png")
    plt.close()


def plot_kpar0_Jan3():

    plot_save_dir = "/home/rbyrne/kpar0_plots_Dec2023"
    calibration_test_filepath = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_cyg_cas_Jan2024"
    diffuse_model_filepath = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_test_diffuse_Jan2024"
    plot_files = [
        f"{calibration_test_filepath}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
        f"{calibration_test_filepath}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_model_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
        f"{diffuse_model_filepath}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_model_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
    ]
    run_names = ["calibrated data", "point source model", "model with diffuse"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for file_ind, filename in enumerate(plot_files):

        data = scipy.io.readsav(filename)["power"]
        k_edges = scipy.io.readsav(filename)["k_edges"]
        power_plot = np.repeat(data, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(
            k_edges_plot,
            power_plot,
            color=colors[file_ind],
            label=run_names[file_ind],
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k-perpendicular (h/Mpc)")
        plt.ylabel("Power")
    plt.legend()
    plt.savefig(f"{plot_save_dir}/kpar0_power_test_diffuse_norm.png")
    plt.close()


def compare_calibration_Jan4():

    plot_save_dir = "/home/rbyrne/kpar0_plots_Dec2023"
    filepath1 = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_cyg_cas_Jan2024"
    filepath2 = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_cyg_cas_Dec2023"
    plot_files = [
        f"{filepath1}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
        f"{filepath2}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
    ]
    run_names = ["new calibration", "old calibration"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for file_ind, filename in enumerate(plot_files):

        data = scipy.io.readsav(filename)["power"]
        k_edges = scipy.io.readsav(filename)["k_edges"]
        power_plot = np.repeat(data, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(
            k_edges_plot,
            power_plot,
            color=colors[file_ind],
            label=run_names[file_ind],
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k-perpendicular (h/Mpc)")
        plt.ylabel("Power")
    plt.legend()
    plt.savefig(f"{plot_save_dir}/kpar0_power_compare_calibration.png")
    plt.close()


def plot_pyuvsim_mmode_Jan17():

    plot_save_dir = "/home/rbyrne/kpar0_plots_Dec2023"
    calibration_test_filepath = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_caltest_cyg_cas_Jan2024"
    diffuse_model_filepath = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_generate_ps_Jan2024"
    plot_files = [
        f"{calibration_test_filepath}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
        f"{calibration_test_filepath}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_model_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
        f"{diffuse_model_filepath}/ps/data/1d_binning/20230819_093023_73MHz__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",
    ]
    run_names = ["calibrated data", "point source model", "model with diffuse"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for file_ind, filename in enumerate(plot_files):

        data = scipy.io.readsav(filename)["power"]
        k_edges = scipy.io.readsav(filename)["k_edges"]
        power_plot = np.repeat(data, 2)
        k_edges_plot = np.concatenate(
            ([k_edges[0]], np.repeat(k_edges[1:-1], 2), [k_edges[-1]])
        )
        plt.plot(
            k_edges_plot,
            power_plot,
            color=colors[file_ind],
            label=run_names[file_ind],
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k-perpendicular (h/Mpc)")
        plt.ylabel("Power")
    plt.legend()
    plt.savefig(f"{plot_save_dir}/kpar0_power_test_pyuvsim_mmode_simulation.png")
    plt.close()


if __name__ == "__main__":

    plot_pyuvsim_mmode_Jan17()
