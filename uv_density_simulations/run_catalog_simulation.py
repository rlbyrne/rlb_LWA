import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvsim
from pyuvsim import mpi
import sys


args = str(sys.argv)
input_uvfits_path = args[0]
catalog_path = args[1]
output_uvfits_path = args[2]

# Read uvfits
uv = pyuvdata.UVData()
uv.read_uvfits(input_uvfits_path)

# Get beam
airy_beam = pyuvsim.AnalyticBeam("airy", diameter=14.)
airy_beam.peak_normalize()

# Read and format diffuse
catalog = pyradiosky.SkyModel()
catalog.read_fhd_catalog(catalog_path)
if not catalog.check():
    print("Error: Catalog fails check.")
# Format catalog to be pyuvsim-compatible
catalog_formatted = pyuvsim.simsetup.SkyModelData(catalog_formatted)
catalog_formatted.share()

# Start MPI
mpi.start_mpi(block_nonroot_stdout=False)

# Run simulation
diffuse_sim_uv = pyuvsim.uvsim.run_uvdata_uvsim(
    input_uv=uv,
    beam_list=BeamList(beam_list=[airy_beam]),
    beam_dict=None,  # Same beam for all ants
    catalog=catalog_formatted,
    quiet=False,
)
