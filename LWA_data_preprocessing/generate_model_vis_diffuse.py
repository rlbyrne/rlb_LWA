# Based on uv_density_simulations/run_diffuse_simulation.py

import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import pyuvsim
from pyuvsim import mpi
import sys
import time


args = sys.argv
diffuse_map_path = args[1]
beam_path = args[2]
input_data_path = args[3]
output_uvfits_path = args[4]

mpi.start_mpi(block_nonroot_stdout=False)
rank = mpi.get_rank()
comm = mpi.world_comm

uv = pyuvdata.UVData()
beam_list = None
diffuse_map_formatted = pyuvsim.simsetup.SkyModelData()

if rank == 0:
    # Read reference data for simulation
    uv.read_uvfits(input_uvfits_path)

    # Get beam
    beam = pyuvdata.UVBeam()
    beam.read(beam_path)
    beam.peak_normalize()
    beam_list = pyuvsim.BeamList(beam_list=[beam])

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
    diffuse_map.jansky_to_kelvin()
    if not diffuse_map.check():
        print("Error: Diffuse map fails check.")
    # Format diffuse map to be pyuvsim-compatible
    diffuse_map_formatted = pyuvsim.simsetup.SkyModelData(diffuse_map)

uv = comm.bcast(uv, root=0)
beam_list = comm.bcast(beam_list, root=0)
diffuse_map_formatted.share(root=0)

# Run simulation
start_time = time.time()
output_uv = pyuvsim.uvsim.run_uvdata_uvsim(
    input_uv=uv,
    beam_list=beam_list,
    beam_dict=None,  # Same beam for all ants
    catalog=diffuse_map_formatted,
    quiet=False,
)
if rank == 0:
    print(f"Simulation time: {(time.time() - start_time)/60.} minutes")
    sys.stdout.flush()
    output_uv.write_uvfits(output_uvfits_path, fix_autos=True)
