from astropy.io import fits
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata


class SkyImage:
    def __init__(self, signal_arr, ra_axis, dec_axis):
        self.signal_arr = signal_arr
        self.ra_axis = ra_axis
        self.dec_axis = dec_axis

    def difference(self, diff_image, tol=1e-5, inplace=False):
        if (
            np.max(np.abs(self.ra_axis - diff_image.ra_axis)) > tol
            or np.max(np.abs(self.dec_axis - diff_image.dec_axis)) > tol
        ):
            print("ERROR: Axes do not match.")
            sys.exit(1)
        diff_signal = self.signal_arr - diff_image.signal_arr
        if inplace:
            self.signal_arr = diff_signal
        else:
            return SkyImage(diff_signal, self.ra_axis, self.dec_axis)

    def plot(
        self,
        x_pixel_extent=None,
        y_pixel_extent=None,
        signal_extent=None,
        diverging_colormap=False,
        colorbar_label="Flux Density (Jy/sr)",
        save_filename=None,
    ):

        if signal_extent is None:
            signal_extent = [np.min(self.signal_arr), np.max(self.signal_arr)]

        use_signal_arr = self.signal_arr
        use_ra_axis = self.ra_axis
        use_dec_axis = self.dec_axis
        x_pixels = np.arange(np.shape(self.signal_arr)[0])
        y_pixels = np.arange(np.shape(self.signal_arr)[1])
        if x_pixel_extent is not None:
            x_pixels = np.intersect1d(
                np.where(x_pixels < np.max(x_pixel_extent)),
                np.where(x_pixels > np.min(x_pixel_extent)),
            )
            use_signal_arr = use_signal_arr[x_pixels, :]
        if y_pixel_extent is not None:
            y_pixels = np.intersect1d(
                np.where(y_pixels < np.max(y_pixel_extent)),
                np.where(y_pixels > np.min(y_pixel_extent)),
            )
            use_signal_arr = use_signal_arr[:, y_pixels]

        if diverging_colormap:
            colormap = "seismic"
        else:
            colormap = "Greys_r"

        fig, ax = plt.subplots()
        plt.imshow(
            use_signal_arr.T,  # imshow plots the 0th axis vertically
            origin="lower",
            interpolation="none",
            cmap=colormap,
            extent=[
                np.min(x_pixels),
                np.max(x_pixels),
                np.min(y_pixels),
                np.max(y_pixels),
            ],
            vmin=signal_extent[0],
            vmax=signal_extent[1],
            aspect="auto",
        )
        plt.axis("equal")
        # ax.set_facecolor('gray')  # make plot background gray
        ax.set_facecolor("black")
        plt.xlabel("RA (pixels)")
        plt.ylabel("Dec. (pixels)")
        # if plot_grid:
        #    plt.grid(which='both', zorder=10, lw=0.5)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=15)
        if save_filename is not None:
            print(f"Saving figure to {save_filename}")
            plt.savefig(save_filename, format="png", dpi=500)
            plt.close()
        else:
            plt.show()


def load_fits(data_filename):

    contents = fits.open(data_filename)
    use_hdu = 0
    signal_arr = np.array(contents[use_hdu].data).T
    signal_arr = np.flip(signal_arr, axis=0)  # RA decreases along the 0th axis

    header = contents[use_hdu].header
    if "CD1_1" in header.keys() and "CD2_2" in header.keys():  # FHD convention
        cdelt1 = header["CD1_1"]
        cdelt2 = header["CD2_2"]
    elif "CDELT1" in header.keys() and "CDELT2" in header.keys():
        cdelt1 = header["CDELT1"]
        cdelt2 = header["CDELT2"]
    else:
        print("ERROR: Header format not recognized.")
        sys.exit(1)

    ra_axis = np.flip(
        header["crval1"] + cdelt1 * (np.arange(header["naxis1"]) - header["crpix1"])
    )
    dec_axis = header["crval2"] + cdelt2 * (
        np.arange(header["naxis2"]) - header["crpix2"]
    )

    return SkyImage(signal_arr, ra_axis, dec_axis)


if __name__=="__main__":

    path = "/Users/ruby/Astro/LWA_pyuvsim_simulations/fhd_rlb_LWA_phasing_sim_Feb2021/output_data"
    save_path = "/Users/ruby/Astro/LWA_pyuvsim_simulations/plots_for_powerpoint"
    true_filename = f"{path}/OVRO-LWA_100ms_sim_center_time_uniform_Dirty_XX.fits"
    unphased_filename = f"{path}/OVRO-LWA_100ms_sim_unphased_uniform_Dirty_XX.fits"
    phased_filename = f"{path}/OVRO-LWA_100ms_sim_phased_uniform_Dirty_XX.fits"
    unphased_1h_filename = f"{path}/OVRO-LWA_1h_sim_unphased_uniform_Dirty_XX.fits"
    phased_1h_filename = f"{path}/OVRO-LWA_1h_sim_phased_uniform_Dirty_XX.fits"

    true = load_fits(true_filename)
    unphased = load_fits(unphased_filename)
    phased = load_fits(phased_filename)
    unphased_1h = load_fits(unphased_1h_filename)
    phased_1h = load_fits(phased_1h_filename)
    unphased_diff = unphased.difference(true)
    phased_diff = phased.difference(true)
    image_span = 100
    x_center = 854
    y_center = 1098
    unphased_diff.plot(
        x_pixel_extent=[x_center - image_span / 2, x_center + image_span / 2],
        y_pixel_extent=[y_center - image_span / 2, y_center + image_span / 2],
        signal_extent=[-1e8, 1e8],
        diverging_colormap=True,
        colorbar_label="Diff. Flux Density (Jy/sr)",
        save_filename=f"{save_path}/10s_unphased_minus_true.png"
    )
    phased_diff.plot(
        x_pixel_extent=[x_center - image_span / 2, x_center + image_span / 2],
        y_pixel_extent=[y_center - image_span / 2, y_center + image_span / 2],
        signal_extent=[-1e8, 1e8],
        diverging_colormap=True,
        colorbar_label="Diff. Flux Density (Jy/sr)",
        save_filename=f"{save_path}/10s_phased_minus_true.png"
    )
    true.plot(
        x_pixel_extent=[x_center - image_span / 2, x_center + image_span / 2],
        y_pixel_extent=[y_center - image_span / 2, y_center + image_span / 2],
        signal_extent=[0, 1e10],
        save_filename=f"{save_path}/10s_true.png"
    )
    unphased.plot(
        x_pixel_extent=[x_center - image_span / 2, x_center + image_span / 2],
        y_pixel_extent=[y_center - image_span / 2, y_center + image_span / 2],
        signal_extent=[0, 1e10],
        save_filename=f"{save_path}/10s_unphased.png"
    )
    image_span_1h = 400
    true.plot(
        x_pixel_extent=[x_center - image_span_1h / 2, x_center + image_span_1h / 2],
        y_pixel_extent=[y_center - image_span_1h / 2, y_center + image_span_1h / 2],
        signal_extent=[0, 1e9],
        save_filename=f"{save_path}/1h_true.png"
    )
    unphased_1h.plot(
        x_pixel_extent=[x_center - image_span_1h / 2, x_center + image_span_1h / 2],
        y_pixel_extent=[y_center - image_span_1h / 2, y_center + image_span_1h / 2],
        signal_extent=[0, 1e9],
        save_filename=f"{save_path}/1h_unphased.png"
    )
    phased_1h.plot(
        x_pixel_extent=[x_center - image_span_1h / 2, x_center + image_span_1h / 2],
        y_pixel_extent=[y_center - image_span_1h / 2, y_center + image_span_1h / 2],
        signal_extent=[0, 1e9],
        save_filename=f"{save_path}/1h_phased.png"
    )
