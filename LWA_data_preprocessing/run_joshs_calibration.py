import pyuvdata
from dsa2000_cal.iterative_calibrator import (
    create_data_input_gen,
    Data,
    IterativeCalibrator,
    DataGenInput,
)
from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
import numpy as np


def data_generator():
    data_ms = "/lustre/rbyrne/2024-03-03/ddcal/20240303_133205_73MHz.ms"
    bright_sources_ms_list = [
        "/lustre/rbyrne/2024-03-03/ddcal/20240303_133205_73MHz_model_Vir.ms",
        "/lustre/rbyrne/2024-03-03/ddcal/20240303_133205_73MHz_model_Cyg.ms",
        "/lustre/rbyrne/2024-03-03/ddcal/20240303_133205_73MHz_model_Cas.ms",
    ]
    background_ms = (
        "/lustre/rbyrne/2024-03-03/ddcal/20240303_133205_73MHz_model_diffuse.ms"
    )

    coherencies = ("XX", "XY", "YX", "YY")

    # Get data
    uv = pyuvdata.UVData()
    uv.read(data_ms)
    uv.select(times=np.min(uv.time_array))
    uv.read(data_ms)
    uv.reorder_pols(order="AIPS")
    uv.reorder_blts()
    uv.reorder_freqs(channel_order="freq")
    uv.phase_to_time(np.mean(uv.time_array))
    vis_data = uv.data_array[np.newaxis, :, :, :]  # Shape Ntimes, Nbls, Nfreqs, Npols
    flags = uv.data_array[np.newaxis, :, :, :]  # Shape Ntimes, Nbls, Nfreqs, Npols
    antenna1 = uv.ant_1_array
    antenna2 = uv.ant_2_array
    times = uv.time_array[0]
    freqs = uv.freq_array

    # Get background
    model = pyuvdata.UVData()
    model.read(background_ms)
    model.select(times=np.min(uv.time_array))
    model.read(data_ms)
    model.reorder_pols(order="AIPS")
    model.reorder_blts()
    model.reorder_freqs(channel_order="freq")
    model.phase_to_time(np.mean(uv.time_array))
    vis_background = model.data_array[
        np.newaxis, np.newaxis, :, :, :
    ]  # Shape 1, Ntimes, Nbls, Nfreqs, Npols

    vis_bright_sources = np.zeros(
        (
            len(bright_sources_ms_list),
            uv.Ntimes,
            uv.Nbls,
            uv.Nfreqs,
            uv.Npols,
        ),
        dtype=complex,
    )
    for file_ind, ms_file in enumerate(bright_sources_ms_list):
        source_model = pyuvdata.UVData()
        source_model.read(ms_file)
        source_model.select(times=np.min(uv.time_array))
        source_model.read(data_ms)
        source_model.reorder_pols(order="AIPS")
        source_model.reorder_blts()
        source_model.reorder_freqs(channel_order="freq")
        source_model.phase_to_time(np.mean(uv.time_array))
        vis_bright_sources[file_ind, :, :, :, :] = source_model.data_array[
            np.newaxis, :, :, :
        ]

    return_data = yield Data(
        sol_int_time_idx=0,
        coherencies=coherencies,
        vis_data=vis_data,
        weights=np.ones_like(vis_data, dtype=float),
        flags=flags,
        vis_bright_sources=vis_bright_sources,
        vis_background=vis_background,
        antenna1=antenna1,
        antenna2=antenna2,
        model_times=times,
        model_freqs=freqs,
        ref_time=times,
    )


if __name__ == "__main__":
    your_generator = data_generator()

    gain_prior_model = GainPriorModel(
        gain_stddev=1.0,
        dd_dof=1,
        di_dof=1,
        double_differential=True,
        dd_type="unconstrained",
        di_type="unconstrained",
        full_stokes=True,
    )
    calibrator = IterativeCalibrator(
        plot_folder="demo_plots",
        run_name="demo",
        gain_probabilistic_model=gain_prior_model,
        full_stokes=True,
        antennas=352,
        verbose=True,  # if using GPU's set to False
        devices=None,
    )
    calibrator.run(your_generator, Ts=None, Cs=None)
