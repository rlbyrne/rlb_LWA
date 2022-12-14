import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import pyuvsim
from pyuvsim import mpi
import sys
import time


args = sys.argv
input_uvfits_path = args[1]
catalog_path = args[2]
output_uvfits_path = args[3]

mpi.start_mpi(block_nonroot_stdout=False)
rank = mpi.get_rank()
comm = mpi.world_comm

uv = pyuvdata.UVData()
beam_list = None
catalog_formatted = pyuvsim.simsetup.SkyModelData()

if rank == 0:
    # Read uvfits
    uv.read_uvfits(input_uvfits_path)

    # Get beam
    airy_beam = pyuvsim.AnalyticBeam("airy", diameter=14.0)
    airy_beam.peak_normalize()
    beam_list = pyuvsim.BeamList(beam_list=[airy_beam])

    # Read and format catalog
    catalog = pyradiosky.SkyModel()
    catalog.read_fhd_catalog(catalog_path)
    if not catalog.check():
        print("Error: Catalog fails check.")
    # For testing, only use a few sources
    #use_inds = np.where(catalog.stokes[0, :, :].value >= 10)[1]
    #catalog.select(component_inds=use_inds)
    # Format catalog to be pyuvsim-compatible
    catalog_formatted = pyuvsim.simsetup.SkyModelData(catalog)

uv = comm.bcast(uv, root=0)
beam_list = comm.bcast(beam_list, root=0)
catalog_formatted.share(root=0)

# Run simulation
start_time = time.time()
output_uv = pyuvsim.uvsim.run_uvdata_uvsim(
    input_uv=uv,
    beam_list=beam_list,
    beam_dict=None,  # Same beam for all ants
    catalog=catalog_formatted,
    quiet=False,
)
if rank == 0:
    print(f"Simulation time: {(time.time() - start_time)/60.} minutes")
    sys.stdout.flush()
    output_uv.write_uvfits(output_uvfits_path)
