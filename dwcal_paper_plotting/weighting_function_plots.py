import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize
from dwcal import delay_weighted_cal as dwcal


def fft_visibilities(uv):
    delay_array = np.fft.fftfreq(uv.Nfreqs, d=uv.channel_width)
    delay_array = np.fft.fftshift(delay_array)
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft(uv.data_array, axis=2), axes=2))
    return fft_abs, delay_array


def calculate_binned_rms(
    vis_array, uvw_array, Nfreqs, nbins=100, min_val=None, max_val=None
):

    bl_lengths = np.sqrt(np.sum(uvw_array**2.0, axis=1))
    if min_val is None:
        min_val = np.min(bl_lengths)
    if max_val is None:
        max_val = np.max(bl_lengths)
    bl_bin_edges = np.linspace(min_val, max_val, num=nbins + 1)
    binned_rms_squared = np.full([nbins, Nfreqs], np.nan, dtype="float")
    for bin_ind in range(nbins):
        bl_inds = np.where(
            (bl_lengths > bl_bin_edges[bin_ind])
            & (bl_lengths <= bl_bin_edges[bin_ind + 1])
        )[0]
        if len(bl_inds) > 0:
            binned_rms_squared[bin_ind, :] = np.mean(
                vis_array[bl_inds, 0, :, 0] ** 2.0, axis=0
            )

    binned_rms = binned_rms_squared**0.5
    return binned_rms, bl_bin_edges


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
    cbar.ax.set_ylabel("Visibility Error RMS (Jy)", rotation=270, labelpad=15)
    plt.xlabel("Baseline Length (m)")
    plt.ylim([-3, 3])
    plt.ylabel("Delay ($\mu$s)")
    plt.title(title)
    if savepath is not None:
        plt.savefig(savepath, dpi=600)
        plt.close()
    else:
        plt.show()


def plot_weighting_function(bin_edges, delay_array):

    wedge_val = 0.141634
    window_val = 10.7401
    wedge_slope_factor =  0.628479
    wedge_delay_buffer = 6.5e-8
    c = 3e8
    weighting_func_delay_vals = np.full((len(bin_edges)-1, len(delay_array)), window_val)
    for bin_ind, bl_length_start in enumerate(bin_edges[:-1]):
        bl_length = (bl_length_start + bin_edges[bin_ind+1])/2.
        wedge_inds = np.where(np.abs(delay_array) <= bl_length*wedge_slope_factor/c + wedge_delay_buffer)[0]
        weighting_func_delay_vals[bin_ind, wedge_inds] = wedge_val

    use_cmap = matplotlib.cm.get_cmap('plasma').copy()
    use_cmap.set_bad(color='whitesmoke')
    plt.imshow(
        weighting_func_delay_vals.T, origin='lower', interpolation='none', cmap=use_cmap,
        extent=[np.min(bin_edges), np.max(bin_edges), np.min(delay_array)*1e6,
                    np.max(delay_array)*1e6],
        aspect='auto',
        norm=matplotlib.colors.LogNorm(vmin=5e-2, vmax=20),
    )
    cbar = plt.colorbar(extend="both")
    cbar.ax.set_ylabel(
        'Weighting Function Value (Jy$^{-2}$)', rotation=270, labelpad=15
    )
    for line_slope in [1.]:
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
    plt.xlabel('Baseline Length (m)')
    plt.ylim([-3, 3])
    plt.ylabel('Delay ($\mu$s)')
    plt.savefig("/Users/ruby/Astro/dwcal_paper_plots/weighting_func.png", dpi=600)
    plt.close()


def plot_model_visibility_error():

    data, model = dwcal.get_test_data(
        model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
        model_use_model=True,
        data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Apr2022",
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
    binned_delay_spec_diff, bin_edges = calculate_binned_rms(
        fft_abs_diff, diff_vis.uvw_array, diff_vis.Nfreqs
    )
    plot_delay_spectra(
        binned_delay_spec_diff,
        bin_edges,
        delay_array,
        title="Model Visibility Error",
        add_lines=[1.0],
        vmin=1e-2,
        vmax=1e1,
        savepath="/Users/ruby/Astro/dwcal_paper_plots/model_vis_error.png",
    )

    plot_weighting_function(bin_edges, delay_array)


if __name__ == "__main__":

    plot_model_visibility_error()
