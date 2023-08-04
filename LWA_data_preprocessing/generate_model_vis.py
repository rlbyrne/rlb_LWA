# Based on uv_density_simulations/run_catalog_simulation.py

import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import pyuvsim
from pyuvsim import mpi
import sys
import time


args = sys.argv
input_data_path = args[1]
output_uvfits_path = args[2]

catalog_path = "../LWA_skymodels/cyg_cas.skyh5"
beam_path = "../LWAbeam_2015.fits"

mpi.start_mpi(block_nonroot_stdout=False)
rank = mpi.get_rank()
comm = mpi.world_comm

uv = pyuvdata.UVData()
beam_list = None
catalog_formatted = pyuvsim.simsetup.SkyModelData()

if rank == 0:
    # Read reference data for simulation
    uv.read(input_data_path)

    # Get beam
    beam = pyuvdata.UVBeam()
    beam.read(beam_path)
    beam.peak_normalize()
    beam_list = pyuvsim.BeamList(beam_list=[beam])

    # Read and format catalog
    catalog = pyradiosky.SkyModel()
    catalog.read_skyh5(catalog_path)
    if not catalog.check():
        print("Error: Catalog fails check.")
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
    output_uv.write_uvfits(output_uvfits_path, fix_autos=True)
