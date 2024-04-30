from astropy.io import fits
import numpy as np
import healpy as hp
import pyradiosky
import astropy.units as units
from astropy.units import Quantity
import sys


def convert_fits_to_pyradiosky(
    fits_filepath,
    freq_mhz,
    output_frame=None,  # Options are "galactic", "equatorial", or None. If None, input frame is preserved.
    output_nside=None,  # If None, input nside is preserved
):

    file_contents = fits.open(fits_filepath)
    map_data = np.array(file_contents[1].data, dtype=float)
    ordering = file_contents[1].header["ORDERING"].lower()
    nside = int(file_contents[1].header["NSIDE"])
    npix = hp.nside2npix(nside)
    coordsys = file_contents[1].header["COORDSYS"]
    file_contents.close()

    history_str = f"From file {fits_filepath.split('/')[-1]}"

    if output_frame is not None:
        # COORDSYS definitions: G = galactic, E = ecliptic, C = celestial = equatorial
        if output_frame == "equatorial":
            output_frame_code = "C"
        elif output_frame == "galactic":
            output_frame_code = "G"
        else:
            sys.exit(
                "ERROR: Unsupported output_frame. Options are equatorial, galactic, or None."
            )

        if coordsys != output_frame_code:  # Transform coordinate frame
            if ordering == "nest":
                nest = True
            elif ordering == "ring":
                nest = False
            else:
                print("WARNING: Unknown ordering. Assuming ring ordering.")
                nest = False
            theta_init, phi_init = hp.pixelfunc.pix2ang(
                nside, np.arange(npix), nest=nest
            )
            rot = hp.rotator.Rotator(coord=[coordsys, output_frame_code])
            theta_rot, phi_rot = rot(theta_init, phi_init)
            map_data = hp.get_interp_val(map_data, theta_rot, phi_rot, nest=nest)
            history_str = f"{history_str}, interpolated to {output_frame} frame"
            coordsys = output_frame_code

    if coordsys == "C":
        frame_str = "icrs"
    elif coordsys == "G":
        frame_str = "galactic"
    else:
        sys.exit("ERROR: Unsupported coordsys.")

    if output_nside is not None and output_nside != nside:  # Interpolate to new nside
        map_data = hp.pixelfunc.ud_grade(
            map_data, output_nside, pess=True, order_in=ordering
        )
        nside = output_nside
        npix = hp.nside2npix(nside)

    skymodel = pyradiosky.SkyModel()
    skymodel.component_type = "healpix"
    skymodel.nside = nside
    skymodel.hpx_order = ordering
    skymodel.frame = frame_str
    skymodel.hpx_frame = frame_str
    skymodel.Nfreqs = 1
    skymodel.Ncomponents = npix
    skymodel.freq_array = Quantity(np.full(skymodel.Nfreqs, freq_mhz * 1e6), "hertz")
    skymodel.stokes = Quantity(
        np.zeros((4, skymodel.Nfreqs, skymodel.Ncomponents)),
        "Kelvin",  # Assume units of Kelvin
    )
    skymodel.stokes[0, 0, :] = map_data * units.Kelvin
    skymodel.hpx_inds = np.arange(skymodel.Ncomponents)
    skymodel.spectral_type = "flat"
    skymodel.history = history_str

    return skymodel


if __name__ == "__main__":

    # Map is downloaded from https://lambda.gsfc.nasa.gov/product/foreground/fg_ovrolwa_radio_maps_get.html

    fits_filepath = (
        "/Users/ruby/Astro/mmode_maps_eastwood/ovro_lwa_sky_map_46.992MHz.fits"
    )
    skymodel = convert_fits_to_pyradiosky(
        fits_filepath,
        46.992,
        output_frame="equatorial",
    )
    skymodel.check()
    skymodel.write_skyh5(
        "/Users/ruby/Astro/mmode_maps_eastwood/ovro_lwa_sky_map_46.992MHz.skyh5",
        run_check=True,
        clobber=True,
    )
    """
    skymodel.kelvin_to_jansky()  # Convert to units Jy/sr
    skymodel.write_skyh5(
        "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial_Jy_per_sr.skyh5",
        run_check=True,
        clobber=True,
    )
    skymodel.stokes *= 4.0 * np.pi / hp.nside2npix(skymodel.nside) # Convert from Jy/sr to Jy/pixel
    skymodel.write_skyh5(
        "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial_Jy_per_pixel.skyh5",
        run_check=True,
        clobber=True,
    )
    """
