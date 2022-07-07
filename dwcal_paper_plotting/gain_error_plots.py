import numpy as np
import pyuvdata
import matplotlib.pyplot as plt


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


def plot_tests_individually():

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


def plot_tests_together():

    cal_unity_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/unity_gains_diagonal.calfits"
    cal_unity_diagonal = pyuvdata.UVCal()
    cal_unity_diagonal.read_calfits(cal_unity_diagonal_path)
    cal_unity_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/unity_gains_dwcal.calfits"
    )
    cal_unity_dwcal = pyuvdata.UVCal()
    cal_unity_dwcal.read_calfits(cal_unity_dwcal_path)

    cal_random_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_initial_gains.calfits"
    cal_random_true = pyuvdata.UVCal()
    cal_random_true.read_calfits(cal_random_true_path)
    cal_random_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_gains_diagonal.calfits"
    cal_random_diagonal = pyuvdata.UVCal()
    cal_random_diagonal.read_calfits(cal_random_diagonal_path)
    cal_random_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_gains_dwcal.calfits"
    )
    cal_random_dwcal = pyuvdata.UVCal()
    cal_random_dwcal.read_calfits(cal_random_dwcal_path)

    cal_ripple_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_initial_gains.calfits"
    cal_ripple_true = pyuvdata.UVCal()
    cal_ripple_true.read_calfits(cal_ripple_true_path)
    cal_ripple_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_gains_diagonal.calfits"
    cal_ripple_diagonal = pyuvdata.UVCal()
    cal_ripple_diagonal.read_calfits(cal_ripple_diagonal_path)
    cal_ripple_dwcal_path = (
        "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_gains_dwcal.calfits"
    )
    cal_ripple_dwcal = pyuvdata.UVCal()
    cal_ripple_dwcal.read_calfits(cal_ripple_dwcal_path)

    plot_gain_error(
        [
            cal_unity_diagonal,
            cal_unity_dwcal,
            cal_random_diagonal,
            cal_random_dwcal,
            cal_ripple_diagonal,
            cal_ripple_dwcal,
        ],
        cal_true_list=[
            None,
            None,
            cal_random_true,
            cal_random_true,
            cal_ripple_true,
            cal_ripple_true,
        ],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/all_tests_comparison.png",
        avg_line_color=[
            "tab:blue",
            "tab:blue",
            "tab:red",
            "tab:red",
            "tab:purple",
            "tab:purple",
        ],
        plot_indiv_ants=False,
        add_lines=[1.0],
    )


if __name__ == "__main__":

    plot_tests_individually()
