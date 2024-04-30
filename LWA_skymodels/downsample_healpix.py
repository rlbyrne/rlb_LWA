import numpy as np
import healpy as hp
from astropy.units import Quantity
import astropy.units as units
import pyradiosky


def downsample_healpix(input_map_path, output_map_path, output_nside, clobber=True):

    diffuse_map = pyradiosky.SkyModel()
    diffuse_map.read_skyh5(input_map_path)

    output_nside = 128
    downsampled_map_data = hp.pixelfunc.ud_grade(
        diffuse_map.stokes[0, 0, :].value,
        output_nside,
        pess=True,
        order_in=diffuse_map.hpx_order,
    )
    diffuse_map.nside = output_nside
    diffuse_map.Ncomponents = hp.nside2npix(output_nside)
    diffuse_map.stokes = Quantity(
        np.zeros((4, diffuse_map.Nfreqs, diffuse_map.Ncomponents)), "Kelvin"
    )
    diffuse_map.stokes[0, 0, :] = downsampled_map_data * units.Kelvin
    diffuse_map.hpx_inds = np.arange(diffuse_map.Ncomponents)
    diffuse_map.check()
    diffuse_map.write_skyh5(
        output_map_path,
        run_check=True,
        clobber=clobber,
    )


if __name__ == "__main__":

    downsample_healpix(
        "/Users/ruby/Astro/mmode_maps_eastwood/ovro_lwa_sky_map_46.992MHz.skyh5",
        "/Users/ruby/Astro/mmode_maps_eastwood/ovro_lwa_sky_map_46.992MHz_nside128.skyh5",
        128,
    )
