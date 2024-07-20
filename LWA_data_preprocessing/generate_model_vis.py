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

uv_list = []
beam_list = None
catalog_formatted = pyuvsim.simsetup.SkyModelData()

if rank == 0:
    # Read reference data for simulation
    uv = pyuvdata.UVData()
    uv.read_ms(input_data_path)
    uv.flag_array[:, :, :] = False
    if uv.telescope_name == "OVRO_MMA":  # Correct telescope location
        uv.telescope_name = "OVRO-LWA"
        uv.set_telescope_params(overwrite=True, warn=True)
    uv.set_uvws_from_antenna_positions(update_vis=False)  # Correct UVWs
    time_list = list(set(uv.time_array))
    uv.phase_to_time(np.mean(time_list))
    uv_list = []
    for use_time in time_list:
        uv_list.append(uv.select(times=use_time, inplace=False))
    # uv.downsample_in_time(n_times_to_avg=uv.Ntimes)

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

mpi.big_bcast(comm, uv_list, root=0)
beam_list = comm.bcast(beam_list, root=0)
catalog_formatted.share(root=0)

# Run simulation
start_time = time.time()
output_uv_list = []
for time_ind in range(len(uv_list)):
    output_uv_single_time = pyuvsim.uvsim.run_uvdata_uvsim(
        input_uv=uv_list[time_ind],
        beam_list=beam_list,
        beam_dict=None,  # Same beam for all ants
        catalog=catalog_formatted,
        quiet=False,
    )
    if time_ind == 0:
        output_uv = output_uv_single_time
    else:
        output_uv.fast_concat(output_uv_single_time, inplace=True)

if rank == 0:
    print(f"Simulation time: {(time.time() - start_time)/60.} minutes")
    sys.stdout.flush()
    # output_uv.write_uvfits(output_uvfits_path, fix_autos=True)
    # Add this for Gregg's simulations:
    output_uv.data_array *= 2
    output_uv.reorder_pols(order="CASA")
    output_uv.write_ms(output_uvfits_path, fix_autos=True)
