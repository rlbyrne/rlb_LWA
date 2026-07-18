# Based on uv_density_simulations/run_catalog_simulation.py

import numpy as np
from astropy.units import Quantity
import pyradiosky
import pyuvdata
import pyuvsim
from pyuvsim import mpi
import sys
import time


def run_pyuvsim_chunk_in_srcs(
    catalog=None,  # SkyModel object or str
    beam_path=None,
    input_data=None,
    output_path=None,
    sources_per_chunk=1000,
):

    use_catalog = pyradiosky.SkyModel()
    use_catalog.read_skyh5(catalog)

    if use_catalog.Ncomponents <= sources_per_chunk:
        output_uv = simulate_with_pyuvsim(
            catalog=use_catalog,  # SkyModel object or str
            beam_path=beam_path,
            input_data=input_data,
        )
        output_uv.write_ms(output_path, fix_autos=True, clobber=True)

    else:
        start_source = 0
        while start_source < use_catalog.Ncomponents:
            end_source = np.min(
                [start_source + sources_per_chunk, use_catalog.Ncomponents]
            )
            cat_chunk = use_catalog.select(
                component_inds=np.arange(start_source, end_source), inplace=False
            )
            output_uv_chunk_new = simulate_with_pyuvsim_simple(
                catalog=cat_chunk,  # SkyModel object or str
                beam_path=beam_path,
                input_data=input_data,
            )
            if start_source == 0:
                output_uv = output_uv_chunk_new
            else:
                output_uv.sum_vis(output_uv_chunk_new, inplace=True)
        output_uv.write_ms(output_path, fix_autos=True, clobber=True)


def run_pyuvsim(
    catalog=None,  # SkyModel object or str
    beam_path=None,
    input_data=None,
    output_path=None,
    freqs_per_chunk=50,
):

    uv = pyuvdata.UVData()
    uv.read(input_data)
    uv.flag_array[:, :, :] = False  # Unflag all
    uv.phase_to_time(np.mean(uv.time_array))

    beam = pyuvdata.UVBeam()
    beam.read(beam_path)
    beam.peak_normalize()
    beam_list = pyuvsim.BeamList(beam_list=[beam])

    cat = pyradiosky.SkyModel()
    cat.read(catalog)
    catalog_formatted = pyuvsim.simsetup.SkyModelData(cat)

    if uv.Nfreqs <= freqs_per_chunk:
        output_uv = pyuvsim.uvsim.run_uvdata_uvsim(
            input_uv=uv,
            beam_list=beam_list,
            beam_dict=None,  # Same beam for all ants
            catalog=catalog_formatted,
            quiet=False,
        )

    else:
        start_freq = 0
        while start_freq < uv.Nfreqs:
            end_freq = np.min([start_freq + freqs_per_chunk, uv.Nfreqs])
            uv_chunk = uv.select(
                freq_chans=np.arange(start_freq, end_freq), inplace=False
            )
            output_uv_chunk_new = pyuvsim.uvsim.run_uvdata_uvsim(
                input_uv=uv_chunk,
                beam_list=beam_list,
                beam_dict=None,  # Same beam for all ants
                catalog=catalog_formatted,
                quiet=False,
            )
            if start_freq == 0:
                output_uv = output_uv_chunk_new
            else:
                output_uv.fast_concat(output_uv_chunk_new, "freq", inplace=True)

    output_uv.data_array *= 2  # Need this for Stokes I convention
    output_uv.phase_center_catalog = (
        uv.phase_center_catalog
    )  # pyuvsim does not preserve phase center info
    output_uv.reorder_pols(order="CASA")
    output_uv.write_ms(output_path, fix_autos=True, clobber=True)


def simulate_with_pyuvsim_simple(
    catalog=None,  # SkyModel object or str
    beam_path=None,
    input_data=None,  # UVData object or str
):

    if isinstance(catalog, str):
        uv = pyuvdata.UVData()
        uv.read(input_data, ignore_single_chan=False)
        uv.flag_array[:, :, :] = False
        uv.phase_to_time(np.mean(uv.time_array))
    else:
        uv = input_data

    if isinstance(catalog, str):
        use_catalog = pyradiosky.SkyModel()
        use_catalog.read(catalog)
    else:
        use_catalog = catalog
    catalog_formatted = pyuvsim.simsetup.SkyModelData(use_catalog)

    beam = pyuvdata.UVBeam()
    beam.read(beam_path)
    beam.peak_normalize()
    beam_list = pyuvsim.BeamList(beam_list=[beam])

    output_uv = pyuvsim.uvsim.run_uvdata_uvsim(
        input_uv=uv,
        beam_list=beam_list,
        beam_dict=None,  # Same beam for all ants
        catalog=catalog_formatted,
        quiet=False,
    )
    output_uv.phase_to_time(np.mean(output_uv.time_array))
    return output_uv


def simulate_with_pyuvsim(
    catalog=None,  # SkyModel object or str
    beam_path=None,
    input_data=None,  # UVData object or str
):

    mpi.start_mpi(block_nonroot_stdout=False)
    rank = mpi.get_rank()
    comm = mpi.world_comm

    uv = pyuvdata.UVData()
    beam_list = None
    catalog_formatted = pyuvsim.simsetup.SkyModelData()

    if rank == 0:
        if isinstance(catalog, str):  # Read reference data for simulation
            uv.read_ms(input_data)
            uv.flag_array[:, :, :] = False
            # uv.set_uvws_from_antenna_positions(update_vis=False)  # Correct UVWs
            uv.phase_to_time(np.mean(uv.time_array))
            # uv.downsample_in_time(n_times_to_avg=uv.Ntimes)

            # if uv.Npols != 4:  # pyuvsim currently only supports data with Npols=4
            #    uv2 = uv.copy()
            #    uv2.polarization_array = [-7, -8]
            #    uv2.data_array[:, :, :] = 0.0 + 1j*0.0  # Zero out crosspol data
            #    uv.fast_concat(uv2, "polarization", inplace=True)
        else:
            uv = input_data

        # Get beam
        beam = pyuvdata.UVBeam()
        beam.read(beam_path)
        beam.peak_normalize()
        beam_list = pyuvsim.BeamList(beam_list=[beam])

        # Read and format catalog
        if isinstance(catalog, str):
            use_catalog = pyradiosky.SkyModel()
            use_catalog.read_skyh5(catalog)
        else:  # Assume catalog is a SkyModel object
            use_catalog = catalog
        if not use_catalog.check():
            print("Error: Catalog fails check.")
        # Format catalog to be pyuvsim-compatible
        catalog_formatted = pyuvsim.simsetup.SkyModelData(use_catalog)

    mpi.big_bcast(comm, uv, root=0)
    # uv = comm.bcast(uv, root=0)
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
        output_uv.data_array *= 2  # Need this for Stokes I convention
        output_uv.phase_center_catalog = (
            uv.phase_center_catalog
        )  # pyuvsim does not preserve phase center info
        output_uv.reorder_pols(order="CASA")
        return output_uv


if __name__ == "__main__":

    args = sys.argv
    catalog = args[1]
    beam_path = args[2]
    input_data = args[3]
    output_path = args[4]

    run_pyuvsim(
        catalog=catalog,
        beam_path=beam_path,
        input_data=input_data,
        output_path=output_path,
    )
