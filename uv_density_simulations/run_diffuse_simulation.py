import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvsim
from pyuvsim import mpi
import sys


args = str(sys.argv)
input_uvfits_path = args[0]
diffuse_map_path = args[1]
output_uvfits_path = args[2]

# Read uvfits
uv = pyuvdata.UVData()
uv.read_uvfits(input_uvfits_path)

# Get beam
airy_beam = pyuvsim.AnalyticBeam("airy", diameter=14.)
airy_beam.peak_normalize()

# Read and format diffuse
diffuse_map = pyradiosky.SkyModel()
diffuse_map.read_skyh5(diffuse_map_path)
# Reformat the map with a spectral index
diffuse_map.spectral_type = "spectral_index"
diffuse_map.spectral_index = np.full(diffuse_map.Ncomponents, -0.8)
diffuse_map.reference_frequency = Quantity(
    np.full(diffuse_map.Ncomponents, diffuse_map.freq_array[0].value), "Hz"
)
diffuse_map._reference_frequency.required = True
diffuse_map.freq_array = None
# Convert map to units of K

if not diffuse_map.check():
    print("Error: Diffuse map fails check.")
# Format diffuse map to be pyuvsim-compatible
diffuse_map_formatted = pyuvsim.simsetup.SkyModelData(diffuse_map)
diffuse_map_formatted.share()

# Start MPI
mpi.start_mpi(block_nonroot_stdout=False)

# Run simulation
diffuse_sim_uv = pyuvsim.uvsim.run_uvdata_uvsim(
    input_uv=uv,
    beam_list=BeamList(beam_list=[airy_beam]),
    beam_dict=None,  # Same beam for all ants
    catalog=diffuse_map_pyuvsim_formatted,
    quiet=False,
)
