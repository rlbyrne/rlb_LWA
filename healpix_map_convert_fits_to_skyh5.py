from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pyradiosky
import astropy.units as units
from astropy.units import Quantity


def convert_fits_to_pyradiosky(fits_filepath, freq_mhz):

    file_contents = fits.open(fits_filepath)
    map_data = np.array(file_contents[1].data, dtype=float)
    ordering = file_contents[1].header["ORDERING"].lower()
    nside = int(file_contents[1].header["NSIDE"])
    file_contents.close()

    skymodel = pyradiosky.SkyModel()
    skymodel.component_type = "healpix"
    skymodel.nside = nside
    skymodel.hpx_order = ordering
    skymodel.Nfreqs = 1
    skymodel.Ncomponents = hp.nside2npix(nside)
    skymodel.freq_array = Quantity(np.full(skymodel.Nfreqs, freq_mhz * 1e6), "hertz")
    skymodel.stokes = Quantity(
        np.zeros((4, skymodel.Nfreqs, skymodel.Ncomponents)), "Kelvin"
    )
    skymodel.stokes[0, 0, :] = map_data * units.Kelvin

    return skymodel


if __name__ == "__main__":
    fits_filepath = (
        "/Users/ruby/Astro/mmode_maps_eastwood/ovro_lwa_sky_map_73.152MHz.fits"
    )
    skymodel = convert_fits_to_pyradiosky(fits_filepath, 73.152)
    skymodel.write_skyh5(
        "/Users/ruby/Astro/mmode_maps_eastwood/ovro_lwa_sky_map_73.152MHz.skyh5",
        run_check=False,
    )
