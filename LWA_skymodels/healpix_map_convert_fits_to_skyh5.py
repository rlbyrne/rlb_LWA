from astropy.io import fits
import numpy as np
import healpy as hp
import pyradiosky
import astropy.units as units
from astropy.units import Quantity


def convert_fits_to_pyradiosky(fits_filepath, freq_mhz):

    file_contents = fits.open(fits_filepath)
    map_data = np.array(file_contents[1].data, dtype=float)
    ordering = file_contents[1].header["ORDERING"].lower()
    nside = int(file_contents[1].header["NSIDE"])
    npix = hp.nside2npix(nside)
    coordsys = file_contents[1].header["COORDSYS"]
    file_contents.close()

    history_str = f"From file {fits_filepath.split("/")[-1]}"

    # Map must be in equatorial coordinates
    # COORDSYS definitions: G = galactic, E = ecliptic, C = celestial = equatorial
    if coordsys != "C":
        if coordsys == "G":  # Remap to equatorial coordinates
            if ordering == "nest":
                nest = True
            elif ordering == "ring":
                nest = False
            else:
                print("WARNING: Unknown ordering. Assuming ring ordering.")
                nest = False
            # Get pixel coordinates
            theta_gal, phi_gal = hp.pixelfunc.pix2ang(nside, np.arange(npix), nest=nest)
            # Convert pixel coordinates
            rot = hp.rotator.Rotator(coord=["C", "G"])
            theta_eq, phi_eq = rot(theta_gal, phi_gal)
            # Interpolate map to new pixel locations
            map_data = hp.get_interp_val(map_data, theta_eq, phi_eq, nest=nest)
            history_str = f"{history_str}, interpolated to equatorial frame"
        else:
            print("WARNING: Unknown coordsys. No reordering applied.")

    skymodel = pyradiosky.SkyModel()
    skymodel.component_type = "healpix"
    skymodel.nside = nside
    skymodel.hpx_order = ordering
    skymodel.frame = "icrs"
    skymodel.Nfreqs = 1
    skymodel.Ncomponents = npix
    skymodel.freq_array = Quantity(np.full(skymodel.Nfreqs, freq_mhz * 1e6), "hertz")
    skymodel.stokes = Quantity(
        np.zeros((4, skymodel.Nfreqs, skymodel.Ncomponents)), "Kelvin"
    )
    skymodel.stokes[0, 0, :] = map_data * units.Kelvin
    skymodel.hpx_inds = np.arange(skymodel.Ncomponents)
    skymodel.spectral_type = "flat"
    skymodel.history = history_str

    return skymodel


if __name__ == "__main__":

    # Map is downloaded from https://lambda.gsfc.nasa.gov/product/foreground/fg_ovrolwa_radio_maps_get.html

    fits_filepath = (
        "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz.fits"
    )
    skymodel = convert_fits_to_pyradiosky(fits_filepath, 73.152)
    skymodel.check()
    skymodel.write_skyh5(
        "/safepool/rbyrne/skymodels/mmode_maps_eastwood/ovro_lwa_sky_map_73.152MHz_equatorial.skyh5",
        run_check=True,
        clobber=True,
    )
