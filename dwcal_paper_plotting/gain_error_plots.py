import numpy as np
import pyuvdata
import matplotlib.pyplot as plt


def plot_gain_error(
    cal_list,
    cal_true_list=None,  # If None, assume true gains are unity
    save_path=None,
    title="",
    avg_line_color=None,
    plot_indiv_ants=True,
    legend_labels=None,
    add_lines=[]
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
        if plot_indiv_ants:
            for ant_ind in range(cal.Nants_data):
                if ant_ind == 0:
                    use_label = "Per Antenna"
                else:
                    use_label = None
                plt.plot(delay_array, np.abs(gains_fft[ant_ind, :]), color="grey", alpha=.3, linewidth=.8, label=use_label)
        mean_gains_fft = np.mean(np.abs(gains_fft), axis=0)
        plt.plot(
            delay_array, mean_gains_fft,
            color=avg_line_color[cal_ind], linewidth=1,
            label=legend_labels[cal_ind]
        )
    ylim = [2e-5, 2e-1]
    plt.plot([0, 0], ylim, color="grey", linestyle="dashed", linewidth=.5)
    for line in add_lines:
        plt.plot([line, line], ylim, color="black", linestyle="dashed", linewidth=.8)
    plt.ylim(ylim)
    plt.yscale("log")
    plt.xlim([np.min(delay_array), np.max(delay_array)])
    plt.xlabel("Delay ($\mu$s)")
    plt.ylabel("Gain Error Amp.")
    plt.legend()
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_tests_individually():

    # Unity gains
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/unity_gains_diagonal.calfits"
    cal_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/unity_gains_dwcal.calfits'

    cal_diagonal = pyuvdata.UVCal()
    cal_diagonal.read_calfits(cal_diagonal_path)
    cal_dwcal = pyuvdata.UVCal()
    cal_dwcal.read_calfits(cal_dwcal_path)

    plot_gain_error(
        [cal_diagonal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_diagonal.png",
        avg_line_color=["tab:cyan"],
        title = "Sky-Based Calibration",
    )
    plot_gain_error(
        [cal_dwcal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_dwcal.png",
        avg_line_color=["tab:red"],
        title = "Delay Weighted Calibration",
    )
    plot_gain_error(
        [cal_diagonal, cal_dwcal],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/unity_gains_comparison.png",
        avg_line_color=["tab:cyan", "tab:red"],
        plot_indiv_ants=False,
        legend_labels = ["Sky-Based Cal", "DWCal"]
    )

    # Randomized gains
    cal_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_initial_gains.calfits"
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_gains_diagonal.calfits"
    cal_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_gains_dwcal.calfits'

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
        avg_line_color=["tab:cyan"],
        title = "Sky-Based Calibration",
    )
    plot_gain_error(
        [cal_dwcal],
        cal_true_list=[cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/random_gains_dwcal.png",
        avg_line_color=["tab:red"],
        title = "Delay Weighted Calibration",
    )
    plot_gain_error(
        [cal_diagonal, cal_dwcal],
        cal_true_list=[cal_true, cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/random_gains_comparison.png",
        avg_line_color=["tab:cyan", "tab:red"],
        plot_indiv_ants=False,
        legend_labels = ["Sky-Based Cal", "DWCal"]
    )

    # Ripple gains
    cal_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_initial_gains.calfits"
    cal_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_gains_diagonal.calfits"
    cal_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_gains_dwcal.calfits'

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
        avg_line_color=["tab:cyan"],
        title = "Sky-Based Calibration",
        add_lines = [1.]
    )
    plot_gain_error(
        [cal_dwcal],
        cal_true_list=[cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/ripple_gains_dwcal.png",
        avg_line_color=["tab:red"],
        title = "Delay Weighted Calibration",
        add_lines = [1.]
    )
    plot_gain_error(
        [cal_diagonal, cal_dwcal],
        cal_true_list=[cal_true, cal_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/ripple_gains_comparison.png",
        avg_line_color=["tab:cyan", "tab:red"],
        plot_indiv_ants=False,
        legend_labels = ["Sky-Based Cal", "DWCal"],
        add_lines = [1.]
    )


def plot_tests_together():

    cal_unity_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/unity_gains_diagonal.calfits"
    cal_unity_diagonal = pyuvdata.UVCal()
    cal_unity_diagonal.read_calfits(cal_unity_diagonal_path)
    cal_unity_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/unity_gains_dwcal.calfits'
    cal_unity_dwcal = pyuvdata.UVCal()
    cal_unity_dwcal.read_calfits(cal_unity_dwcal_path)

    cal_random_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_initial_gains.calfits"
    cal_random_true = pyuvdata.UVCal()
    cal_random_true.read_calfits(cal_random_true_path)
    cal_random_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_gains_diagonal.calfits"
    cal_random_diagonal = pyuvdata.UVCal()
    cal_random_diagonal.read_calfits(cal_random_diagonal_path)
    cal_random_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/random_gains_dwcal.calfits'
    cal_random_dwcal = pyuvdata.UVCal()
    cal_random_dwcal.read_calfits(cal_random_dwcal_path)

    cal_ripple_true_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_initial_gains.calfits"
    cal_ripple_true = pyuvdata.UVCal()
    cal_ripple_true.read_calfits(cal_ripple_true_path)
    cal_ripple_diagonal_path = "/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_gains_diagonal.calfits"
    cal_ripple_diagonal = pyuvdata.UVCal()
    cal_ripple_diagonal.read_calfits(cal_ripple_diagonal_path)
    cal_ripple_dwcal_path = '/Users/ruby/Astro/dwcal_tests_Jun2022/caltest_May19/ripple_gains_dwcal.calfits'
    cal_ripple_dwcal = pyuvdata.UVCal()
    cal_ripple_dwcal.read_calfits(cal_ripple_dwcal_path)

    plot_gain_error(
        [cal_unity_diagonal, cal_unity_dwcal, cal_random_diagonal, cal_random_dwcal, cal_ripple_diagonal, cal_ripple_dwcal],
        cal_true_list=[None, None, cal_random_true, cal_random_true, cal_ripple_true, cal_ripple_true],
        save_path="/Users/ruby/Astro/dwcal_paper_plots/all_tests_comparison.png",
        avg_line_color=["tab:cyan", "tab:cyan", "tab:red", "tab:red", "tab:purple", "tab:purple"],
        plot_indiv_ants=False,
        add_lines = [1.]
    )


if __name__ == "__main__":

    plot_tests_individually()
