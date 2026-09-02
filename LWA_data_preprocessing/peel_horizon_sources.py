import pyuvdata
import numpy as np
import bdsf
from calico import calibration_wrappers
from astropy.io import fits
from astropy.wcs import WCS


def get_pixel_az_za(x_pix, y_pix, header):

    wcs_cel = WCS(header).celestial
    ny, nx = header["NAXIS2"], header["NAXIS1"]

    crpix1, crpix2 = header["CRPIX1"] - 1, header["CRPIX2"] - 1
    cd = wcs_cel.pixel_scale_matrix

    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    dx = x_pix - crpix1
    dy = y_pix - crpix2
    xi = cd[0, 0] * dx + cd[0, 1] * dy
    eta = cd[1, 0] * dx + cd[1, 1] * dy

    az = np.degrees(np.arctan2(eta, -xi)) % 360.0

    R = 180.0 / np.pi
    r = np.sqrt(xi**2 + eta**2)
    za = np.degrees(np.arcsin(r * np.pi / 180))

    return az, za


def get_horizon_sources(fits_path):

    img = bdsf.process_image(
        fits_path,
        thresh_pix=10,  # Default is 5, increase to get only the brightest sources
    )

    x_pixels = np.full(len(img.gaussians), np.nan)
    y_pixels = np.full(len(img.gaussians), np.nan)
    total_flux = np.full(len(img.gaussians), np.nan)
    for source_ind, source in enumerate(img.gaussians):
        x_pixels[source_ind] = source.centre_pix[0]
        y_pixels[source_ind] = source.centre_pix[1]
        total_flux[source_ind] = source.total_flux

    with fits.open(fits_path) as hdul:
        header = hdul[0].header
    az_vals, za_vals = get_pixel_az_za(x_pixels, y_pixels, header)
    horizon_source_inds = np.where((za_vals > 85) + ~np.isfinite(za_vals))

    az_vals, za_vals = get_pixel_az_za(
        x_pixels[horizon_source_inds], y_pixels[horizon_source_inds], header
    )
    horizon_fluxes = total_flux[horizon_source_inds]

    return az_vals, horizon_fluxes


def simulate_horizon_sources(uv, source_azimuths, source_fluxes):

    Nsources = len(source_azimuths)
    proj_baseline = -1 * uv.uvw_array[:, [0]] * np.cos(
        np.deg2rad(source_azimuths[np.newaxis, :])
    ) + uv.uvw_array[:, [1]] * np.sin(
        np.deg2rad(source_azimuths[np.newaxis, :])
    )  # Shape (Nbls, Nsources)
    wls = 3e8 / uv.freq_array
    baseline_fringe = source_fluxes[np.newaxis, np.newaxis, :] * np.exp(
        2
        * np.pi
        * 1j
        * proj_baseline[:, np.newaxis, :]
        / wls[np.newaxis, :, np.newaxis]
    )  # Shape (Nbls, Nfreqs, Nsources)

    uv_list = []
    for source_ind in range(Nsources):
        uv_new = uv.copy()
        uv_new.data_array[:, :, 0] = baseline_fringe[:, :, source_ind]
        uv_new.data_array[:, :, 1] = baseline_fringe[:, :, source_ind]
        uv_new.data_array[:, :, 2] = 0 + 0 * 1j
        uv_new.data_array[:, :, 3] = 0 + 0 * 1j
        uv_new.flag_array[...] = False
        uv_new.uvw_array[:, 2] = 0

        uv_list.append(uv_new)

    return uv_list

def peel_sources(uv, source_sim_list):

    data_peeled, uvcal_list = calibration_wrappers.peeling_wrapper(
        uv,
        source_sim_list,
        gain_init_to_vis_ratio=False,
        gain_init_stddev=0.1,
        min_cal_baseline_lambda=10,
        max_cal_baseline_lambda=125,
        verbose=True,
        max_source_offset_deg=None,
        source_offset_taper_deg=0.25,
        parallel=True,
        n_workers=5,
        lambda_val=0,
    )
    return data_peeled

if __name__ == "__main__":

    data_path = "/fast/rbyrne/20260419_055641-055832_44MHz_wsclean_selfcal_deep_flagging_tmp_dir/20260419_055641-055832_calico_peeled.ms"
    fits_path = "/fast/rbyrne/20260419_055641-055832_44MHz_wsclean_selfcal_deep_flagging_tmp_dir/20260419_055641-055832_calico_peeled-dirty.fits"
    output_path = "/fast/rbyrne/20260419_055641-055832_44MHz_wsclean_selfcal_deep_flagging_tmp_dir/20260419_055641-055832_calico_horizon_peeled.ms"
    source_azimuths, source_fluxes = get_horizon_sources(fits_path)
    uv = pyuvdata.UVData()
    uv.read(data_path, ignore_single_chan=False)
    uv_list = simulate_horizon_sources(uv, source_azimuths, source_fluxes)
    data_peeled = peel_sources(uv, uv_list)
    data_peeled.write_ms(output_path)