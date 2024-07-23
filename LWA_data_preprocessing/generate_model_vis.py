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
catalog_path = args[1]
beam_path = args[2]
input_data_path = args[3]
output_uvfits_path = args[4]

mpi.start_mpi(block_nonroot_stdout=False)
rank = mpi.get_rank()
comm = mpi.world_comm

uv = pyuvdata.UVData()
beam_list = None
catalog_formatted = pyuvsim.simsetup.SkyModelData()

if rank == 0:
    # Read reference data for simulation
    uv.read_ms(input_data_path)
    uv.flag_array[:, :, :] = False
    if uv.telescope_name == "OVRO_MMA":  # Correct telescope location
        uv.telescope_name = "OVRO-LWA"
        uv.set_telescope_params(overwrite=True, warn=True)
    uv.set_uvws_from_antenna_positions(update_vis=False)  # Correct UVWs
    uv.phase_to_time(np.mean(uv.time_array))
    # uv.downsample_in_time(n_times_to_avg=uv.Ntimes)

    # if uv.Npols != 4:  # pyuvsim currently only supports data with Npols=4
    #    uv2 = uv.copy()
    #    uv2.polarization_array = [-7, -8]
    #    uv2.data_array[:, :, :] = 0.0 + 1j*0.0  # Zero out crosspol data
    #    uv.fast_concat(uv2, "polarization", inplace=True)

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

# mpi.big_bcast(comm, uv, root=0)
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
    # output_uv.write_uvfits(output_uvfits_path, fix_autos=True)
    # Add this for Gregg's simulations:
    output_uv.data_array *= 2
    output_uv.phase_center_catalog = (
        uv.phase_center_catalog
    )  # pyuvsim does not preserve phase center info
    output_uv.reorder_pols(order="CASA")
    output_uv.write_ms(output_uvfits_path, fix_autos=True)
