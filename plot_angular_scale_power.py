import numpy as np
import matplotlib.pyplot as plt
import pyuvdata


def get_binned_power(uvdata, xmin=None, xmax=None, nbins=50):

    bl_lengths = np.sqrt(np.sum(uvdata.uvw_array**2.0, axis=1))
    if xmin is None:
        xmin = np.min(bl_lengths)
    if xmax is None:
        xmax = np.max(bl_lengths)
    bl_bin_edges = np.linspace(xmin, xmax, num=nbins + 1)

    binned_power = np.full([nbins], np.nan, dtype="float")
    for bin_ind in range(nbins):
        bl_inds = np.where(
            (bl_lengths > bl_bin_edges[bin_ind])
            & (bl_lengths <= bl_bin_edges[bin_ind + 1])
        )[0]
        if len(bl_inds) > 0:
            binned_power[bin_ind] = np.nanmean(
                np.abs(uvdata.data_array[bl_inds, 0, :, 0]) ** 2.0
            )

    return bl_bin_edges, binned_power


def plot_power(data_path, model_path, plot_save_dir, plot_prefix=""):

    data = pyuvdata.UVData()
    data.read(data_path)
    data.data_array[np.where(data.flag_array)] = np.nan
    data.filename = [""]
    model = pyuvdata.UVData()
    model.read(model_path)
    model.data_array[np.where(model.flag_array)] = np.nan
    model.filename = [""]
    residual = data.sum_vis(
        model,
        difference=True,
        inplace=False,
        override_params=[
            "scan_number_array",
            "flag_array",
            "phase_center_id_array",
            "phase_center_app_ra",
            "phase_center_app_dec",
            "phase_center_frame_pa",
            "phase_center_catalog",
            "filename",
            "nsample_array",
        ]
    )

    bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
    bl_bin_edges, data_power = get_binned_power(
        data, xmin=0, xmax=np.max(bl_lengths), nbins=50
    )
    bl_bin_edges, model_power = get_binned_power(
        model, xmin=0, xmax=np.max(bl_lengths), nbins=50
    )
    bl_bin_edges, residual_power = get_binned_power(
        residual, xmin=0, xmax=np.max(bl_lengths), nbins=50
    )

    legend_names = ["data", "model", "residual"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    bin_edges_plot = np.concatenate(
        ([bl_bin_edges[0]], np.repeat(bl_bin_edges[1:-1], 2), [bl_bin_edges[-1]])
    )
    for plot_ind, power in enumerate([data_power, model_power, residual_power]):
        power_plot = np.repeat(power, 2)
        plt.plot(
            bin_edges_plot,
            power_plot,
            color=colors[plot_ind],
            label=legend_names[plot_ind],
        )

    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlim([np.min(bl_bin_edges), np.max(bl_bin_edges)])
    plt.ylim([0, np.nanmax(power_plot)])
    plt.xlabel("Baseline Length (m)")
    plt.ylabel("Power")

    plt.legend()
    plt.savefig(f"{plot_save_dir}/{plot_prefix}_power.png")
    plt.close()

    fractional_power_recovered = (1 - residual_power/data_power) * 100.0
    power_plot = np.repeat(fractional_power_recovered, 2)
    plt.plot(
        bin_edges_plot,
        power_plot,
        color=colors[0],
    )

    plt.xlim([np.min(bl_bin_edges), np.max(bl_bin_edges)])
    plt.ylim([-10, 100])
    plt.xlabel("Baseline Length (m)")
    plt.ylabel("Fractional Power Recovered (%)")
    plt.savefig(f"{plot_save_dir}/{plot_prefix}_frac_power_recovered.png")
    plt.close()

if __name__=="__main__":

    model_files = [
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/orig_model.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_VLSS_sim.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_deGasperin_sources_sim.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_deGasperin_cyg_cas_sim.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_deGasperin_cyg_cas_sim_NMbeam.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_conj_mmode_with_cyg_cas_pyuvsim_nside128_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_with_cyg_cas_matvis_nside128_sim.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_conj_mmode_with_cyg_cas_matvis_nside512_sim.ms",
    ]
    data_files = [
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_orig.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_VLSS.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_deGasperin_sources.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_deGasperin_cyg_cas.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_deGasperin_cyg_cas_NMbeam.ms",
        "/data03/rbyrne/20231222/test_pyuvsim_modeling/cal46_time11_newcal_mmode_with_cyg_cas_pyuvsim_nside128.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_newcal_mmode_with_cyg_cas_matvis_nside128.ms",
        "/data03/rbyrne/20231222/matvis_modeling/cal46_time11_newcal_mmode_with_cyg_cas_matvis_nside512.ms",
    ]
    model_names = [
        "orig",
        "VLSS",
        "deGasperin_sources",
        "deGasperin_cyg_cas",
        "deGasperin_cyg_cas_NMbeam",
        "mmode_pyuvsim_nside128",
        "mmode_matvis_nside128",
        "mmode_matvis_nside512",
    ]
    for file_ind in range(len(model_names)):
        plot_power(
            data_files[file_ind],
            model_files[file_ind],
            "/data03/rbyrne/20231222/test_pyuvsim_modeling/power_plots",
            plot_prefix=model_names[file_ind]
        )
