import numpy as np
import matplotlib.pyplot as plt
import pyuvdata


def get_binned_power(uvdata, use_pols=None, xmin=None, xmax=None, nbins=50):

    bl_lengths = np.sqrt(np.sum(uvdata.uvw_array**2.0, axis=1))
    if xmin is None:
        xmin = np.min(bl_lengths)
    if xmax is None:
        xmax = np.max(bl_lengths)
    bl_bin_edges = np.linspace(xmin, xmax, num=nbins + 1)

    if use_pols is None:
        use_pols = uvdata.polarization_array
    pol_inds = np.array(np.where(np.in1d(uvdata.polarization_array, use_pols))[0])

    binned_power = np.full([nbins], np.nan, dtype="float")
    for bin_ind in range(nbins):
        bl_inds = np.where(
            (bl_lengths > bl_bin_edges[bin_ind])
            & (bl_lengths <= bl_bin_edges[bin_ind + 1])
        )[0]
        if len(bl_inds) > 0:
            binned_power[bin_ind] = np.nanmean(
                np.abs(
                    uvdata.data_array[
                        bl_inds[:, np.newaxis, np.newaxis, np.newaxis],
                        0,
                        :,
                        pol_inds[np.newaxis, np.newaxis, np.newaxis, :],
                    ]
                )
                ** 2.0
            )

    return bl_bin_edges, binned_power


def plot_power(data_path, model_path, plot_save_dir, plot_prefix=""):

    data = pyuvdata.UVData()
    model = pyuvdata.UVData()
    data.read(data_path)
    model.read(model_path)
    data.data_array[np.where(data.flag_array)] = np.nan
    model.data_array[np.where(model.flag_array)] = np.nan
    data.filename = [""]
    model.filename = [""]
    data.conjugate_bls()
    model.conjugate_bls()
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))
    data.reorder_blts()
    model.reorder_blts()
    data.reorder_pols(order="AIPS")
    model.reorder_pols(order="AIPS")
    data.reorder_freqs(channel_order="freq")
    model.reorder_freqs(channel_order="freq")
    # model.data_array = np.conj(model.data_array)

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
            "lst_array",
            "telescope_location",
            "instrument",
            "integration_time",
            "telescope_name",
            "antenna_diameters",
            "uvw_array",
            "rdate",
            "gst0",
            "earth_omega",
            "dut1",
            "timesys",
        ],
    )

    bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))

    pol_names = ["XX", "YY", "cross"]
    use_pols = [[-5], [-6], [-7, -8]]

    for pol_ind in range(len(pol_names)):
        bl_bin_edges, data_power = get_binned_power(
            data, use_pols=use_pols[pol_ind], xmin=0, xmax=np.max(bl_lengths), nbins=50
        )
        bl_bin_edges, model_power = get_binned_power(
            model, use_pols=use_pols[pol_ind], xmin=0, xmax=np.max(bl_lengths), nbins=50
        )
        bl_bin_edges, residual_power = get_binned_power(
            residual,
            use_pols=use_pols[pol_ind],
            xmin=0,
            xmax=np.max(bl_lengths),
            nbins=50,
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

        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlim([np.min(bl_bin_edges), np.max(bl_bin_edges)])
        if pol_names[pol_ind] == "cross":
            plt.ylim([0, 1e7])
        else:
            plt.ylim([0, 1.3e8])
        plt.xlabel("Baseline Length (m)")
        plt.ylabel("Power")

        plt.legend()
        plt.savefig(f"{plot_save_dir}/{plot_prefix}_{pol_names[pol_ind]}_power.png")
        plt.close()

        fractional_power_recovered = (1 - residual_power / data_power) * 100.0
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
        plt.savefig(
            f"{plot_save_dir}/{plot_prefix}_{pol_names[pol_ind]}_frac_power_recovered.png"
        )
        plt.close()


if __name__ == "__main__":

    model_files = [
        "/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_conj_mmode_46.992MHz_nside512_sim.uvfits",
    ]
    data_files = [
        "/data03/rbyrne/20231222/test_diffuse_normalization/cal46_time11_newcal_deGasperin_cyg_cas_48MHz_residual.ms",
    ]
    model_names = [
        "mmode_with_residual_data",
    ]
    for file_ind in range(len(model_names)):
        plot_power(
            data_files[file_ind],
            model_files[file_ind],
            "/data03/rbyrne/20231222/test_diffuse_normalization/power_plots",
            plot_prefix=model_names[file_ind],
        )
