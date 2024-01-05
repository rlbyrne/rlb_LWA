import numpy as np
import healpy as hp
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import pyuvsim


diffuse_map_path = "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial.skyh5"
diffuse_map = pyradiosky.SkyModel()
diffuse_map.read_skyh5(diffuse_map_path)
print(diffuse_map.nside)

use_nside = 512
downsampled_map_data = hp.pixelfunc.ud_grade(
    diffuse_map.stokes[0, 0, :].value, use_nside, pess=True, order_in=diffuse_map.ordering
)
diffuse_map.nside = use_nside
diffuse_map.Ncomponents = hp.nside2npix(use_nside)
diffuse_map.stokes = Quantity(
    np.zeros((4, diffuse_map.Nfreqs, diffuse_map.Ncomponents)), "Kelvin"
)
diffuse_map.stokes[0, 0, :] = downsampled_map_data * units.Kelvin
diffuse_map.hpx_inds = np.arange(diffuse_map.Ncomponents)
diffuse_map.check()
diffuse_map.write_skyh5(
    "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial_nside512.skyh5",
    run_check=True,
    clobber=True,
)
