from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib


class SkyImage:
    def __init__(
        self, signal_arr, ra_axis=None, dec_axis=None, x_range=None, y_range=None
    ):
        self.signal_arr = signal_arr
        self.ra_axis = ra_axis
        self.dec_axis = dec_axis
        if x_range is None:
            self.x_range = np.array([0, np.shape(signal_arr)[0] - 1])
        else:
            self.x_range = x_range
        if y_range is None:
            self.y_range = np.array([0, np.shape(signal_arr)[1] - 1])
        else:
            self.y_range = y_range

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

    def crop_image(self, new_x_range=None, new_y_range=None, inplace=False):

        use_signal_arr = self.signal_arr
        x_pixels = np.arange(
            np.min(self.x_range), np.max(self.x_range) + 1, 1, dtype=int
        )
        y_pixels = np.arange(
            np.min(self.y_range), np.max(self.y_range) + 1, 1, dtype=int
        )
        if new_x_range is not None:
            x_pixels = np.intersect1d(
                np.where(x_pixels <= np.max(new_x_range)),
                np.where(x_pixels >= np.min(new_x_range)),
            )
            use_signal_arr = use_signal_arr[x_pixels, :]
            use_x_range = np.array(
                [
                    np.max([np.min(x_pixels), np.min(new_x_range)]),
                    np.min([np.max(x_pixels), np.max(new_x_range)]),
                ]
            )
        else:
            use_x_range = self.x_range
        if new_y_range is not None:
            y_pixels = np.intersect1d(
                np.where(y_pixels <= np.max(new_y_range)),
                np.where(y_pixels >= np.min(new_y_range)),
            )
            use_signal_arr = use_signal_arr[:, y_pixels]
            use_y_range = np.array(
                [
                    np.max([np.min(y_pixels), np.min(new_y_range)]),
                    np.min([np.max(y_pixels), np.max(new_y_range)]),
                ]
            )
        else:
            use_y_range = self.y_range

        if inplace:
            self.signal_arr = use_signal_arr
            self.x_range = use_x_range
            self.y_range = use_y_range
        else:
            return SkyImage(use_signal_arr, x_range=use_x_range, y_range=use_y_range)

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

        self.crop_image(
            new_x_range=x_pixel_extent, new_y_range=y_pixel_extent, inplace=True
        )

        if diverging_colormap:
            colormap = "seismic"
        else:
            colormap = "Greys_r"

        fig, ax = plt.subplots()
        plt.imshow(
            self.signal_arr.T,  # imshow plots the 0th axis vertically
            origin="lower",
            interpolation="none",
            cmap=colormap,
            extent=[
                np.min(self.x_range),
                np.max(self.x_range),
                np.min(self.y_range),
                np.max(self.y_range),
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
    signal_arr = np.array(
        contents[use_hdu].data
    ).T  # Not sure why this needs to be transposed
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

    return SkyImage(signal_arr, ra_axis=ra_axis, dec_axis=dec_axis)


def plot_fits_file(
    data_filename,
    edge_crop_ratio=0.2,
    signal_extent=None,
    colorbar_label="Surface Brightness (Jy/sr)",
    save_filename=None,
    title=None,
    mark_pointing_ctr=False,
    symlog = False,
):

    hdu = fits.open(data_filename)[0]
    wcs = WCS(hdu.header)
    plot_data = hdu.data
    if hdu.header["NAXIS"] == 2:
        use_slices = ["x", "y"]
        plot_coord_1 = 0
        plot_coord_2 = 1
    elif hdu.header["NAXIS"] == 4:
        plot_data = plot_data[0, 0, :, :]
        use_slices = [0,0,"x","y"]
        plot_coord_1 = 2
        plot_coord_2 = 3
    else:
        print("ERROR: Unknown format.")
        sys.exit(1)

    if signal_extent is None:
        vmin = None
        vmax = None
        if symlog:
            linthresh = (np.max(plot_data)-np.min(plot_data))*1e-3
    else:
        vmin = np.min(signal_extent)
        vmax = np.max(signal_extent)
        if symlog:
            linthresh = (vmax-vmin)*1e-3

    if symlog:
        norm = matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    plt.subplot(projection=wcs, slices=use_slices)
    plt.imshow(
        plot_data,
        origin="lower",
        cmap="inferno",
        interpolation=None,
        norm=norm,
    )
    plt.grid(color="white", ls="solid", lw=0.2)

    n_x_pixels = np.shape(hdu.data)[plot_coord_1]
    n_y_pixels = np.shape(hdu.data)[plot_coord_2]
    plt.xlim(
        [
            int(np.round((1 - edge_crop_ratio) * n_x_pixels)),
            int(np.round(edge_crop_ratio * n_x_pixels)),
        ]
    )
    plt.ylim(
        [
            int(np.round(edge_crop_ratio * n_y_pixels)),
            int(np.round((1 - edge_crop_ratio) * n_y_pixels)),
        ]
    )
    if mark_pointing_ctr:
        plt.plot(n_x_pixels/2., n_y_pixels/2., '+', color='white', markersize=5, lw=0.1)

    plt.xlabel("RA")
    plt.ylabel("Dec.")
    if title is not None:
        plt.title(title)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=10)

    if save_filename is not None:
        print(f"Saving figure to {save_filename}")
        plt.savefig(save_filename, format="png", dpi=500)
        plt.close()
    else:
        plt.show()
