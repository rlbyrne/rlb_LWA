import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.optimize
from dwcal import delay_weighted_cal as dwcal


def fft_visibilities(uv):
    delay_array = np.fft.fftfreq(uv.Nfreqs, d=uv.channel_width)
    delay_array = np.fft.fftshift(delay_array)
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft(uv.data_array, axis=2), axes=2))
    fft_abs *= uv.channel_width  # Normalize FFT
    return fft_abs, delay_array


def calculate_binned_variance(
    vis_array, uvw_array, Nfreqs, nbins=300, min_val=None, max_val=None
):

    bl_lengths = np.sqrt(np.sum(uvw_array**2.0, axis=1))
    if min_val is None:
        min_val = np.min(bl_lengths)
    if max_val is None:
        max_val = np.max(bl_lengths)
    bl_bin_edges = np.linspace(min_val, max_val, num=nbins + 1)
    binned_variance = np.full([nbins, Nfreqs], np.nan, dtype="float")
    for bin_ind in range(nbins):
        bl_inds = np.where(
            (bl_lengths > bl_bin_edges[bin_ind])
            & (bl_lengths <= bl_bin_edges[bin_ind + 1])
        )[0]
        if len(bl_inds) > 0:
            binned_variance[bin_ind, :] = np.mean(
                vis_array[bl_inds, 0, :, 0] ** 2.0, axis=0
            )

    return binned_variance, bl_bin_edges


def plot_delay_spectra(
    binned_delay_spec,
    bin_edges,
    delay_array,
    title="",
    add_lines=[],
    vmin=None,
    vmax=None,
    c=3e8,
    savepath=None,
):

    use_cmap = matplotlib.cm.get_cmap("plasma").copy()
    use_cmap.set_bad(color="whitesmoke")
    if vmin is not None:
        if vmin < 0:
            norm = matplotlib.colors.SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if np.min(binned_delay_spec) < 0:
            norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.LogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
    plt.imshow(
        binned_delay_spec.T,
        origin="lower",
        interpolation="none",
        cmap=use_cmap,
        norm=norm,
        extent=[
            np.min(bin_edges),
            np.max(bin_edges),
            np.min(delay_array) * 1e6,
            np.max(delay_array) * 1e6,
        ],
        aspect="auto",
    )

    for line_slope in add_lines:
        plt.plot(
            [np.min(bin_edges), np.max(bin_edges)],
            [
                np.min(bin_edges) / c * line_slope * 1e6,
                np.max(bin_edges) / c * line_slope * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
        plt.plot(
            [np.min(bin_edges), np.max(bin_edges)],
            [
                -np.min(bin_edges) / c * line_slope * 1e6,
                -np.max(bin_edges) / c * line_slope * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )

    cbar = plt.colorbar(extend="both")
    cbar.ax.set_ylabel(
        "Visibility Variance (Jy$^{2}$/s$^2$)", rotation=270, labelpad=15
    )
    plt.xlabel("Baseline Length (m)")
    plt.ylim([np.min(delay_array) * 1e6, np.max(delay_array) * 1e6])
    plt.ylabel("Delay ($\mu$s)")
    plt.title(title)
    if savepath is not None:
        plt.savefig(savepath, dpi=600)
        plt.close()
    else:
        plt.show()


def plot_weighting_function(bin_edges, delay_array, vmin=1e-1, vmax=1e3):

    wedge_val = 0.141634
    window_val = 10.7401
    wedge_slope_factor = 0.628479
    wedge_delay_buffer = 6.5e-8
    c = 3e8
    weighting_func_delay_vals = np.full(
        (len(bin_edges) - 1, len(delay_array)), window_val
    )
    for bin_ind, bl_length_start in enumerate(bin_edges[:-1]):
        bl_length = (bl_length_start + bin_edges[bin_ind + 1]) / 2.0
        wedge_inds = np.where(
            np.abs(delay_array)
            <= bl_length * wedge_slope_factor / c + wedge_delay_buffer
        )[0]
        weighting_func_delay_vals[bin_ind, wedge_inds] = wedge_val

    use_cmap = matplotlib.cm.get_cmap("plasma").copy()
    use_cmap.set_bad(color="whitesmoke")
    plt.imshow(
        weighting_func_delay_vals.T,
        origin="lower",
        interpolation="none",
        cmap=use_cmap,
        extent=[
            np.min(bin_edges),
            np.max(bin_edges),
            np.min(delay_array) * 1e6,
            np.max(delay_array) * 1e6,
        ],
        aspect="auto",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    cbar = plt.colorbar(extend="both")
    cbar.ax.set_ylabel("Visibility Variance (Jy$^{2}$)", rotation=270, labelpad=15)
    for line_slope in [1.0]:
        plt.plot(
            [np.min(bin_edges), np.max(bin_edges)],
            [
                np.min(bin_edges) / c * line_slope * 1e6,
                np.max(bin_edges) / c * line_slope * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
        plt.plot(
            [np.min(bin_edges), np.max(bin_edges)],
            [
                -np.min(bin_edges) / c * line_slope * 1e6,
                -np.max(bin_edges) / c * line_slope * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
    plt.xlabel("Baseline Length (m)")
    plt.ylim([-3, 3])
    plt.ylabel("Delay ($\mu$s)")
    plt.savefig("/Users/ruby/Astro/dwcal_paper_plots/weighting_func.png", dpi=600)
    plt.close()


def plot_weighting_function_exponential(
    bin_edges, delay_array, channel_width, vmin=1e9, vmax=1e13
):

    wedge_slope_factor_inner = 0.23
    wedge_slope_factor_outer = 0.7
    wedge_delay_buffer = 6.5e-8
    wedge_variance_inner = 1084.9656666412166
    wedge_variance_outer = 28.966173241799588
    window_min_variance = 5.06396954e-01
    window_exp_amp = 1.19213736e03
    window_exp_width = 6.93325643e-08

    c = 3e8

    exp_function = (
        window_exp_amp * np.exp(-np.abs(delay_array) / window_exp_width / 2)
        + window_min_variance
    )
    exp_function[
        np.where(exp_function > wedge_variance_outer)[0]
    ] = wedge_variance_outer
    weighting_func_delay_vals = np.repeat(
        exp_function[np.newaxis, :], len(bin_edges) - 1, axis=0
    )

    bl_lengths = np.array(
        [
            (bin_edges[bin_ind] + bin_edges[bin_ind + 1]) / 2.0
            for bin_ind in range(len(bin_edges) - 1)
        ]
    )
    for delay_ind, delay_val in enumerate(delay_array):
        wedge_bls_outer = np.where(
            wedge_slope_factor_outer * bl_lengths / c + wedge_delay_buffer
            > np.abs(delay_val)
        )[0]
        weighting_func_delay_vals[wedge_bls_outer, delay_ind] = wedge_variance_outer
        wedge_bls_inner = np.where(
            wedge_slope_factor_inner * bl_lengths / c + wedge_delay_buffer
            > np.abs(delay_val)
        )[0]
        weighting_func_delay_vals[wedge_bls_inner, delay_ind] = wedge_variance_inner

    weighting_func_delay_vals *= (
        channel_width**2
    )  # Make normalization conform with delay spectra

    use_cmap = matplotlib.cm.get_cmap("plasma").copy()
    use_cmap.set_bad(color="whitesmoke")
    plt.imshow(
        weighting_func_delay_vals.T,
        origin="lower",
        interpolation="none",
        cmap=use_cmap,
        extent=[
            np.min(bin_edges),
            np.max(bin_edges),
            np.min(delay_array) * 1e6,
            np.max(delay_array) * 1e6,
        ],
        aspect="auto",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    cbar = plt.colorbar(extend="both")
    cbar.ax.set_ylabel(
        "Visibility Variance (Jy$^{2}$/s$^2$)", rotation=270, labelpad=15
    )
    for line_slope in [1.0]:
        plt.plot(
            [np.min(bin_edges), np.max(bin_edges)],
            [
                np.min(bin_edges) / c * line_slope * 1e6,
                np.max(bin_edges) / c * line_slope * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
        plt.plot(
            [np.min(bin_edges), np.max(bin_edges)],
            [
                -np.min(bin_edges) / c * line_slope * 1e6,
                -np.max(bin_edges) / c * line_slope * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
    plt.xlabel("Baseline Length (m)")
    plt.ylim([np.min(delay_array) * 1e6, np.max(delay_array) * 1e6])
    plt.ylabel("Delay ($\mu$s)")
    plt.savefig(
        "/Users/ruby/Astro/dwcal_paper_plots/weighting_func_exponential.png", dpi=600
    )
    plt.close()


def plot_model_visibility_error():

    data, model = dwcal.get_test_data(
        model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_bright_sources_Jun2022",
        # model_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_bright_sources_Jun2022",
        model_use_model=True,
        data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Jun2022",
        # data_path="/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Jun2022",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        debug_limit_freqs=None,
        use_antenna_list=None,
        use_flagged_baselines=False,
    )

    diff_vis = data.diff_vis(model, inplace=False)

    fft_abs_diff, delay_array = fft_visibilities(diff_vis)
    binned_delay_spec_diff, bin_edges = calculate_binned_variance(
        fft_abs_diff, diff_vis.uvw_array, diff_vis.Nfreqs
    )
    plot_delay_spectra(
        binned_delay_spec_diff,
        bin_edges,
        delay_array,
        title="",
        add_lines=[1.0],
        vmin=1e9,
        vmax=1e13,
        savepath="/Users/ruby/Astro/dwcal_paper_plots/model_vis_error_Jun2022.png",
        # savepath="/home/rbyrne/model_vis_error.png"
    )

    plot_weighting_function_exponential(bin_edges, delay_array, diff_vis.channel_width)


def plot_example_baseline_weights():

    bl_lengths = [2500, 500, 5]

    Nfreqs = 384
    channel_width = 80000.0
    min_freq_mhz = 167075000.0 / 1e6
    max_freq_mhz = 197715000.0 / 1e6
    c = 3e8

    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
        color=sns.color_palette("CMRmap_r", len(bl_lengths))
    )
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    delay_array = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=channel_width))

    # Simple wedge exclusion
    # wedge_val = 0.141634
    # window_val = 10.7401
    # wedge_slope_factor = 0.628479
    # wedge_delay_buffer = 6.5e-8
    # delay_weighting = np.full((len(bl_lengths), Nfreqs), window_val)
    # for bl_ind, bl_len in enumerate(bl_lengths):
    #    wedge_delays = np.where(
    #        np.abs(delay_array) < wedge_slope_factor * bl_len / c + wedge_delay_buffer
    #    )[0]
    #    delay_weighting[bl_ind, wedge_delays] = wedge_val

    # Exponential weighting function
    wedge_slope_factor_inner = 0.23
    wedge_slope_factor_outer = 0.7
    wedge_delay_buffer = 6.5e-8
    wedge_variance_inner = 1084.9656666412166
    wedge_variance_outer = 28.966173241799588
    window_min_variance = 5.06396954e-01
    window_exp_amp = 1.19213736e03
    window_exp_width = 6.93325643e-08
    exp_function = (
        window_exp_amp * np.exp(-np.abs(delay_array) / window_exp_width / 2)
        + window_min_variance
    )
    exp_function[
        np.where(exp_function > wedge_variance_outer)[0]
    ] = wedge_variance_outer
    delay_weighting = np.repeat(exp_function[np.newaxis, :], len(bl_lengths), axis=0)

    for delay_ind, delay_val in enumerate(delay_array):
        wedge_bls_outer = np.where(
            wedge_slope_factor_outer * np.array(bl_lengths) / c + wedge_delay_buffer
            > np.abs(delay_val)
        )[0]
        delay_weighting[wedge_bls_outer, delay_ind] = wedge_variance_outer
        wedge_bls_inner = np.where(
            wedge_slope_factor_inner * np.array(bl_lengths) / c + wedge_delay_buffer
            > np.abs(delay_val)
        )[0]
        delay_weighting[wedge_bls_inner, delay_ind] = wedge_variance_inner
    delay_weighting = 1.0 / (delay_weighting * channel_width**2.0)

    linewidth = 1.2
    delta_linewidth = 0
    for bl_ind in range(len(bl_lengths)):
        ax[0].plot(
            delay_array * 1e6,
            delay_weighting[bl_ind, :],
            linewidth=linewidth,
            label=f"{bl_lengths[bl_ind]} m",
        )
        linewidth -= delta_linewidth
    ax[0].set_xlim(np.min(delay_array) * 1e6, np.max(delay_array) * 1e6)
    ax[0].set_yscale("log")
    ax[0].set_ylim(5e-4 / (8e4) ** 2, 4 / (8e4) ** 2)
    ax[0].set_xlabel("Delay ($\mu$s)")
    ax[0].set_ylabel("Weighting Function Value (s$^2$/Jy$^2$)")
    ax[0].grid()
    ax[0].legend(title="Baseline Length")

    freq_weighting = np.fft.ifft(
        np.fft.ifftshift(delay_weighting * channel_width**2.0, axes=1), axis=1
    )
    freq_weighting_shifted = np.fft.fftshift(freq_weighting, axes=1)
    freqs = np.array([channel_width * freq_ind for freq_ind in range(Nfreqs)])
    freqs = np.fft.fftshift(freqs)
    freqs[np.where(freqs > freqs[-1])[0]] -= np.max(freqs) + channel_width
    linewidth = 1.2
    delta_linewidth = 0
    for bl_ind in range(len(bl_lengths)):
        ax[1].plot(
            freqs / 1e6,
            np.real(freq_weighting_shifted[bl_ind, :]),
            linewidth=linewidth,
            label=f"{bl_lengths[bl_ind]} m",
        )
        linewidth -= delta_linewidth
    ax[1].set_xlim(np.min(freqs) / 1e6, np.max(freqs) / 1e6)
    ax[1].set_ylim(-0.5, 2)
    ax[1].set_xlabel("$\Delta$ Frequency (MHz)")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].set_ylabel("Weighting Function Value (Jy$^{-2}$)")
    ax[1].grid()
    ax[1].legend(title="Baseline Length")

    plt.savefig(
        "/Users/ruby/Astro/dwcal_paper_plots/weighting_func_select_bls.png", dpi=600
    )
    plt.close()

    plot_mat_bl_inds = [2, 1, 0]
    fig, ax = plt.subplots(nrows=1, ncols=len(plot_mat_bl_inds), figsize=(16, 6))
    subplot_ind = 0
    for bl in plot_mat_bl_inds:
        weight_mat = np.zeros((Nfreqs, Nfreqs), dtype=complex)
        for freq_ind1 in range(Nfreqs):
            for freq_ind2 in range(Nfreqs):
                weight_mat[freq_ind1, freq_ind2] = freq_weighting[
                    bl, np.abs(freq_ind1 - freq_ind2)
                ]
        use_cmap = matplotlib.cm.get_cmap("Spectral_r").copy()
        cax = ax[subplot_ind].imshow(
            np.real(weight_mat).T,
            origin="lower",
            interpolation="none",
            cmap=use_cmap,
            extent=[
                np.min(min_freq_mhz),
                np.max(max_freq_mhz),
                np.min(min_freq_mhz),
                np.max(max_freq_mhz),
            ],
            aspect="auto",
            norm=matplotlib.colors.SymLogNorm(linthresh=1e-3, vmin=-1, vmax=1),
        )
        ax[subplot_ind].set_xlabel("Frequency 1 (MHz)")
        ax[subplot_ind].set_ylabel("Frequency 2 (MHz)")
        ax[subplot_ind].set_title(
            f"Weighting Matrix, Baseline Length {bl_lengths[bl]} m"
        )
        ax[subplot_ind].set_aspect("equal")
        subplot_ind += 1

    cbar = fig.colorbar(cax, ax=ax.ravel().tolist(), extend="both")
    cbar.set_label("Weighting Matrix Value (Jy$^{-2}$)", rotation=270, labelpad=15)
    plt.savefig("/Users/ruby/Astro/dwcal_paper_plots/weighting_mats.png", dpi=600)
    plt.close()


if __name__ == "__main__":

    plot_example_baseline_weights()
