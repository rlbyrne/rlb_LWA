import pyuvsim
import pyuvdata
import pyradiosky
import numpy as np
#import matplotlib.pyplot as plt
import sys
from pyuvsim.telescope import BeamList
from astropy.units import Quantity


uv = pyuvdata.UVData()
uv.read_uvfits("/safepool/rbyrne/mwa_data/1061316296.uvfits")

airy_beam = pyuvsim.AnalyticBeam("airy", diameter=14.)
airy_beam.peak_normalize()

healpix_map_path = "/safepool/rbyrne/diffuse_map.skyh5"
diffuse_map = pyradiosky.SkyModel()
print("Reading map")
diffuse_map.read_skyh5(healpix_map_path)
# Reformat the map a bit to be compatible with pyuvsim
diffuse_map.spectral_type = "spectral_index"
diffuse_map.spectral_index = np.full(diffuse_map.Ncomponents, -0.8)
diffuse_map.reference_frequency = Quantity(
    np.full(diffuse_map.Ncomponents, diffuse_map.freq_array[0].value), "Hz"
)
diffuse_map_formatted = pyuvsim.simsetup.SkyModelData(sky_in=diffuse_map)
#diffuse_map.freq_array = None
print("Starting diffuse simulation")
diffuse_sim_uv = pyuvsim.uvsim.run_uvdata_uvsim(
    input_uv=uv,
    beam_list=BeamList(beam_list=[airy_beam]),
    beam_dict=None,  # Same beam for all ants
    catalog=diffuse_map,
    quiet=False,
)
