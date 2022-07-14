import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats


def diff_gains(cal1, cal2):

    cal_diff = cal1.copy()
    if cal2 is None:
        cal_diff.gain_array -= 1.
    else:
        cal1_ant_names = cal1.antenna_names[cal1.ant_array]
        cal2_ant_names = cal2.antenna_names[cal2.ant_array]
        for ant_ind, ant_name in enumerate(cal1_ant_names):
            gains = cal1.gain_array[ant_ind, :, :, :, :]
            true_gains = cal2.gain_array[
                np.where(cal2_ant_names == ant_name)[0][0], :, :, :, :
            ]
            cal_diff.gain_array[ant_ind, :, :, :, :] = gains - true_gains

    return cal_diff


def get_median_range(vals, frac_included=0.5):

    elements = np.shape(vals)[0]
    channels = np.shape(vals)[1]
    result = np.zeros((channels, 2), dtype=float)
    for chan_ind in range(channels):
        vals_sorted = np.sort(vals[:, chan_ind])
        start_pos = (1.0 - frac_included) / 2.0 * elements
        end_pos = elements - start_pos
        start_val = (start_pos - np.floor(start_pos)) * vals_sorted[
            int(np.floor(start_pos))
        ] + (np.floor(start_pos) + 1 - start_pos) * vals_sorted[
            int(np.floor(start_pos) + 1)
        ]
        end_val = (end_pos - np.floor(end_pos)) * vals_sorted[
            int(np.floor(end_pos))
        ] + (np.floor(end_pos) + 1 - end_pos) * vals_sorted[int(np.floor(end_pos) + 1)]
        result[chan_ind, 0] = start_val
        result[chan_ind, 1] = end_val

    return result


def plot_gain_error_frequency(
    cal_list,
    cal_true_list=None,  # If None, assume true gains are unity
    save_path=None,
    title="",
    avg_line_color=None,
    plot_indiv_ants=True,
    legend_labels=None,
    add_lines=[],
    plot_distribution=False,
):

    if avg_line_color is None:
        avg_line_color = np.full(len(cal_list), "black", dtype=object)
    if legend_labels is None:
        legend_labels = np.full(len(cal_list), "Average", dtype=object)

    for cal_ind, cal in enumerate(cal_list):
        if cal_true_list is None:
            cal_error = diff_gains(cal, None)
        else:
            cal_true = cal_true_list[cal_ind]
            cal_error = diff_gains(cal, cal_true)
        freq_array = cal_error.freq_array[0, :] / 1e6
        if plot_indiv_ants:
            for ant_ind in range(cal.Nants_data):
                if ant_ind == 0:
                    use_label = "Per Antenna"
                else:
                    use_label = None
                plt.plot(
                    freq_array,
                    np.abs(cal_error.gain_array[ant_ind, 0, :, 0, 0]),
                    color="grey",
                    alpha=0.3,
                    linewidth=0.8,
                    label=use_label,
                )
        mean_gains = np.mean(np.abs(cal_error.gain_array[:, 0, :, 0, 0]), axis=0)
        if plot_distribution:
            median_range = get_median_range(np.abs(cal_error.gain_array[:, 0, :, 0, 0]))
            plt.fill_between(
                freq_array,
                median_range[:, 0],
                median_range[:, 1],
                color=avg_line_color[cal_ind],
                alpha=0.2,
                linewidth=0,
            )
        plt.plot(
            freq_array,
            mean_gains,
            color=avg_line_color[cal_ind],
            linewidth=1,
            label=legend_labels[cal_ind],
        )
    ylim = [0, 0.0045]
    plt.ylim(ylim)
    # plt.yscale("log")
    plt.grid(linewidth=0.3)
    plt.xlim([np.min(freq_array), np.max(freq_array)])
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Gain Error Amp.")
    plt.legend()
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_gain_error(
    cal_list,
    cal_true_list=None,  # If None, assume true gains are unity
    save_path=None,
    title="",
    avg_line_color=None,
    plot_indiv_ants=True,
    legend_labels=None,
    add_lines=[],
    plot_distribution=False,
):

    if avg_line_color is None:
        avg_line_color = np.full(len(cal_list), "black", dtype=object)
    if legend_labels is None:
        legend_labels = np.full(len(cal_list), "Average", dtype=object)

    for cal_ind, cal in enumerate(cal_list):
        cal_error = cal.copy()
        if cal_true_list is None:
            cal_error.gain_array -= 1.0
        else:
            cal_true = cal_true_list[cal_ind]
            if cal_true is None:
                cal_error.gain_array -= 1.0
            else:
                cal_ant_names = cal.antenna_names[cal.ant_array]
                true_ant_names = cal_true.antenna_names[cal_true.ant_array]
                for ant_ind, ant_name in enumerate(cal_ant_names):
                    gains = cal.gain_array[ant_ind, :, :, :, :]
                    true_gains = cal_true.gain_array[
                        np.where(true_ant_names == ant_name)[0][0], :, :, :, :
                    ]
                    cal_error.gain_array[ant_ind, :, :, :, :] = gains - true_gains

        delay_array = np.fft.fftfreq(cal_error.Nfreqs, d=cal_error.channel_width)
        delay_array = np.fft.fftshift(delay_array)
        delay_array *= 1e6  # Convert to units microseconds
        gains_fft = np.fft.fftshift(
            np.fft.fft(cal_error.gain_array[:, 0, :, 0, 0], axis=1), axes=1
        )
        gains_fft *= cal_error.channel_width  # Normalize FFT
        if plot_indiv_ants:
            for ant_ind in range(cal.Nants_data):
                if ant_ind == 0:
                    use_label = "Per Antenna"
                else:
                    use_label = None
                plt.plot(
                    delay_array,
                    np.abs(gains_fft[ant_ind, :]),
                    color="grey",
                    alpha=0.3,
                    linewidth=0.8,
                    label=use_label,
                )
        if plot_distribution:
            median_range = get_median_range(np.abs(gains_fft))
            plt.fill_between(
                delay_array,
                median_range[:, 0],
                median_range[:, 1],
                color=avg_line_color[cal_ind],
                alpha=0.2,
                linewidth=0,
            )
        mean_gains_fft = np.mean(np.abs(gains_fft), axis=0)
        plt.plot(
            delay_array,
            mean_gains_fft,
            color=avg_line_color[cal_ind],
            linewidth=1,
            label=legend_labels[cal_ind],
        )
    ylim = [0, 3.5e4]
    for line in add_lines:
        plt.plot([line, line], ylim, color="black", linestyle="dashed", linewidth=0.5)
    plt.ylim(ylim)
    plt.grid(linewidth=0.3)
    plt.xlim([np.min(delay_array), np.max(delay_array)])
    plt.xlabel("Delay ($\mu$s)")
    plt.ylabel("Gain Error Amp. (s$^{-2}$)")
    plt.legend()
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_gain_errors():

    # Unity gains
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun16/unity_gains_diagonal.calfits"
    # cal_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun6/unity_gains_dwcal.calfits'
    cal_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/unity_gains_dwcal.calfits"
    )

    cal_diagonal = pyuvdata.UVCal()
    cal_diagonal.read_calfits(cal_diagonal_path)
    cal_dwcal = pyuvdata.UVCal()
    cal_dwcal.read_calfits(cal_dwcal_path)

    plot_gain_error(
        [cal_diagonal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_diagonal.png",
        avg_line_color=["tab:blue"],
        title="Sky-Based Calibration",
    )
    plot_gain_error_frequency(
        [cal_diagonal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_diagonal_freq.png",
        avg_line_color=["tab:blue"],
        title="Sky-Based Calibration",
    )
    plot_gain_error(
        [cal_dwcal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_dwcal.png",
        avg_line_color=["tab:red"],
        title="Delay Weighted Calibration",
    )
    plot_gain_error_frequency(
        [cal_dwcal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_dwcal_freq.png",
        avg_line_color=["tab:red"],
        title="Delay Weighted Calibration",
    )
    plot_gain_error(
        [cal_diagonal, cal_dwcal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_comparison.png",
        avg_line_color=["tab:blue", "tab:red"],
        plot_indiv_ants=False,
        legend_labels=["Sky-Based Cal", "DWCal"],
        plot_distribution=True,
    )
    plot_gain_error_frequency(
        [cal_diagonal, cal_dwcal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_comparison_freq.png",
        avg_line_color=["tab:blue", "tab:red"],
        plot_indiv_ants=False,
        legend_labels=["Sky-Based Cal", "DWCal"],
        plot_distribution=True,
    )

    # Randomized gains
    cal_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/random_initial_gains.calfits"
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/random_gains_diagonal.calfits"
    cal_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/random_gains_dwcal.calfits"
    )

    cal_true = pyuvdata.UVCal()
    cal_true.read_calfits(cal_true_path)
    cal_diagonal = pyuvdata.UVCal()
    cal_diagonal.read_calfits(cal_diagonal_path)
    cal_dwcal = pyuvdata.UVCal()
    cal_dwcal.read_calfits(cal_dwcal_path)

    plot_gain_error(
        [cal_diagonal],
        cal_true_list=[cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/random_gains_diagonal.png",
        avg_line_color=["tab:blue"],
        title="Sky-Based Calibration",
    )
    plot_gain_error(
        [cal_dwcal],
        cal_true_list=[cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/random_gains_dwcal.png",
        avg_line_color=["tab:red"],
        title="Delay Weighted Calibration",
    )
    plot_gain_error(
        [cal_diagonal, cal_dwcal],
        cal_true_list=[cal_true, cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/random_gains_comparison.png",
        avg_line_color=["tab:blue", "tab:red"],
        plot_indiv_ants=False,
        legend_labels=["Sky-Based Cal", "DWCal"],
        plot_distribution=True,
    )
    plot_gain_error_frequency(
        [cal_diagonal, cal_dwcal],
        cal_true_list=[cal_true, cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/random_gains_comparison_freq.png",
        avg_line_color=["tab:blue", "tab:red"],
        plot_indiv_ants=False,
        legend_labels=["Sky-Based Cal", "DWCal"],
        plot_distribution=True,
    )

    # Ripple gains
    cal_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/ripple_initial_gains.calfits"
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/ripple_gains_diagonal.calfits"
    cal_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/ripple_gains_dwcal.calfits"
    )

    cal_true = pyuvdata.UVCal()
    cal_true.read_calfits(cal_true_path)
    cal_diagonal = pyuvdata.UVCal()
    cal_diagonal.read_calfits(cal_diagonal_path)
    cal_dwcal = pyuvdata.UVCal()
    cal_dwcal.read_calfits(cal_dwcal_path)

    plot_gain_error(
        [cal_diagonal],
        cal_true_list=[cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/ripple_gains_diagonal.png",
        avg_line_color=["tab:blue"],
        title="Sky-Based Calibration",
        add_lines=[1.0],
    )
    plot_gain_error(
        [cal_dwcal],
        cal_true_list=[cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/ripple_gains_dwcal.png",
        avg_line_color=["tab:red"],
        title="Delay Weighted Calibration",
        add_lines=[1.0],
    )
    plot_gain_error(
        [cal_diagonal, cal_dwcal],
        cal_true_list=[cal_true, cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/ripple_gains_comparison.png",
        avg_line_color=["tab:blue", "tab:red"],
        plot_indiv_ants=False,
        legend_labels=["Sky-Based Cal", "DWCal"],
        add_lines=[1.0],
        plot_distribution=True,
    )
    plot_gain_error_frequency(
        [cal_diagonal, cal_dwcal],
        cal_true_list=[cal_true, cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/ripple_gains_comparison_freq.png",
        avg_line_color=["tab:blue", "tab:red"],
        plot_indiv_ants=False,
        legend_labels=["Sky-Based Cal", "DWCal"],
        plot_distribution=True,
    )


def produce_kde(data, xvals, yvals):

    data_real = np.real(data).flatten()
    data_imag = np.imag(data).flatten()
    rvs = np.append(data_real[:, np.newaxis], data_imag[:, np.newaxis], axis=1)

    kde = scipy.stats.kde.gaussian_kde(rvs.T)

    # Regular grid to evaluate KDE upon
    x, y = np.meshgrid(xvals, yvals)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    kde_vals = kde(grid_coords.T)
    kde_vals = kde_vals.reshape(len(xvals), len(yvals))

    percent_vals = np.zeros_like(kde_vals)
    kde_total = np.sum(kde_vals)
    running_total = 0.0
    # Sort values in reverse order
    for val in np.sort(kde_vals.flatten())[::-1]:
        percent_vals[np.where(kde_vals == val)] = running_total / kde_total
        running_total += val

    return kde_vals, percent_vals


def histogram_plot_2d(
    plot_data,
    plot_range=0.03,
    nbins=50,
    colorbar_range=None,
    plot_contours=True,
    savepath=None,
    title="",
    axis_label="Gain",
    center_on_one=False,
    mark_center=False,
):

    matplotlib.rcParams.update({'font.size': 14})

    bins_imag = np.linspace(-plot_range, plot_range, num=nbins + 1)
    if center_on_one:
        bins_real = bins_imag + 1
    else:
        bins_real = bins_imag

    hist, x_edges, y_edges = np.histogram2d(
        np.real(plot_data).flatten(),
        np.imag(plot_data).flatten(),
        bins=[bins_real, bins_imag],
    )
    #hist /= np.sum(hist)
    if plot_contours:
        kde, percent_plot = produce_kde(plot_data, bins_real, bins_imag)

    if colorbar_range is None:
        colorbar_range = [0, np.max(hist)]

    plt.figure(figsize=[7, 5.5])
    plt.imshow(
        hist.T,
        interpolation="none",
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        cmap="inferno",
        vmin=colorbar_range[0],
        vmax=colorbar_range[1],
    )
    cbar = plt.colorbar(extend="max")
    cbar.ax.set_ylabel("Histogram Count", rotation=270, labelpad=20)
    if plot_contours:
        plt.contour(
            bins_real,
            bins_imag,
            percent_plot,
            levels=[0.5, 0.9],
            colors="white",
            linestyles=["solid", "dashed"],
            linewidths=.7,
        )
    if mark_center:
        plt.scatter([1], [0], marker="P", color="white", edgecolors="black", s=150)
    plt.xlabel("{}, Real Part".format(axis_label))
    # plt.xticks(rotation=45)
    plt.ylabel("{}, Imaginary Part".format(axis_label))
    plt.title(title)
    plt.tight_layout()  # Ensure that axes labels don't get cut off
    plt.grid(linewidth=0.2, color="white")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600)
    plt.close()


def plot_gain_error_hists():

    # Randomized gains
    cal_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/random_initial_gains.calfits"
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/random_gains_diagonal.calfits"
    cal_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_Jun17/random_gains_dwcal.calfits"
    )

    cal_true = pyuvdata.UVCal()
    cal_true.read_calfits(cal_true_path)
    cal_diagonal = pyuvdata.UVCal()
    cal_diagonal.read_calfits(cal_diagonal_path)
    cal_dwcal = pyuvdata.UVCal()
    cal_dwcal.read_calfits(cal_dwcal_path)

    histogram_plot_2d(
        cal_true.gain_array.flatten(),
        plot_range=0.03,
        nbins=50,
        plot_contours=True,
        savepath="/Users/ruby/Astro/dwcal_paper_plots/true_gains_hist.png",
        title="",
        axis_label="Gain Value",
        center_on_one=True,
        mark_center=False
    )

    cal_diag_error = diff_gains(cal_diagonal, cal_true)
    histogram_plot_2d(
        cal_diag_error.gain_array.flatten(),
        plot_range=0.005,
        nbins=50,
        plot_contours=True,
        colorbar_range=[0,350],
        savepath="/Users/ruby/Astro/dwcal_paper_plots/diagonal_gains_hist.png",
        title="Sky-Based Calibration",
        axis_label="Gain Error",
        center_on_one=False,
        mark_center=False
    )

    cal_dwcal_error = diff_gains(cal_dwcal, cal_true)
    histogram_plot_2d(
        cal_dwcal_error.gain_array.flatten(),
        plot_range=0.005,
        nbins=50,
        plot_contours=True,
        colorbar_range=[0,350],
        savepath="/Users/ruby/Astro/dwcal_paper_plots/dwcal_gains_hist.png",
        title="DWCal",
        axis_label="Gain Error",
        center_on_one=False,
        mark_center=False
    )


if __name__ == "__main__":

    plot_gain_error_hists()
