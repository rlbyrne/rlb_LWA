import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import pyuvsim
from pyuvsim import mpi
import sys


args = str(sys.argv)
input_uvfits_path = args[0]
catalog_path = args[1]
output_uvfits_path = args[2]

mpi.start_mpi(block_nonroot_stdout=False)
rank = mpi.get_rank()
comm = mpi.get_comm()

uv = pyuvdata.UVData()
beam_list = None
catalog = pyradiosky.SkyModel()

if rank == 0:
    # Read uvfits
    uv.read_uvfits(input_uvfits_path)

    # Get beam
    airy_beam = pyuvsim.AnalyticBeam("airy", diameter=14.0)
    airy_beam.peak_normalize()
    beam_list = BeamList(beam_list=[airy_beam])

    # Read and format catalog
    catalog.read_fhd_catalog(catalog_path)
    if not catalog.check():
        print("Error: Catalog fails check.")
    # Format catalog to be pyuvsim-compatible
    catalog_formatted = pyuvsim.simsetup.SkyModelData(catalog_formatted)

uv = comm.bcast(uv, root=0)
beam_list = comm.bcast(beam_list, root=0)
catalog_formatted.share(root=0)

# Run simulation
diffuse_sim_uv = pyuvsim.uvsim.run_uvdata_uvsim(
    input_uv=uv,
    beam_list=beam_list,
    beam_dict=None,  # Same beam for all ants
    catalog=catalog_formatted,
    quiet=False,
)
