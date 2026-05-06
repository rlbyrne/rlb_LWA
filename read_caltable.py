from typing import Optional, List
import os
import subprocess
import sys
import shutil
import numpy as np
import pyuvdata
import datetime
from generate_model_vis_fftvis import run_fftvis_sim
import smooth_cal_solutions
from calico import calibration_wrappers, calibration_qa
import casacore.tables as tbl


def read_caltable_safely2(
    table_path, example_data  # UVData or str containing data file path
):
    """
    This function replaces pyuvdata's UVCal.read() function due to a bug in that function.
    See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/1648.
    Deprecated function with resolution of that issue!
    """

    with tbl.table(table_path, readonly=True, ack=False) as tb:
        if "SPECTRAL_WINDOW_ID" not in tb.colnames():
            cal = pyuvdata.UVCal()
            cal.read(table_path)
            return cal

        main_data = tb.getcol("CPARAM")
        main_flags = tb.getcol("FLAG")
        ant_ids = tb.getcol("ANTENNA1")
        spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")

    # 2. Get Metadata from sub-tables (one read each)
    with tbl.table(table_path + "/ANTENNA", readonly=True, ack=False) as tb:
        ant_names = tb.getcol("NAME")
        n_ants = len(ant_names)

    with tbl.table(table_path + "/OBSERVATION", readonly=True, ack=False) as tb:
        # Assuming the first row's time range is representative
        mjd_seconds = np.mean(tb.getcol("TIME_RANGE")[0])
        jd = (mjd_seconds / 86400.0) + 2400000.5

    with tbl.table(table_path + "/SPECTRAL_WINDOW", readonly=True, ack=False) as tb:
        all_freqs_list = tb.getcol("CHAN_FREQ")  # Usually a list of arrays
        # Flatten all frequencies into one master frequency axis
        frequencies = np.concatenate(all_freqs_list)
        n_chan_total = len(frequencies)

    # 3. Organize the data using NumPy indexing
    # We assume N_times = 1 for Bandpass, or consistent across SPWs
    unique_ants = np.unique(ant_ids)
    unique_spws = np.unique(spw_ids)

    # Identify dimensions (Pol, Chan, Time)
    # CPARAM shape is usually [Rows, Channels, Polarizations]
    n_pols = main_data.shape[2]
    # We need to determine n_times based on rows per antenna
    n_rows_per_ant = np.sum(ant_ids == unique_ants[0])
    n_times = n_rows_per_ant // len(unique_spws)
    print(n_times)

    # Pre-allocate arrays: [Ant, Time, Total_Freq, Pol]
    final_data = np.zeros((n_ants, n_times, n_chan_total, n_pols), dtype=complex)
    final_flags = np.ones((n_ants, n_times, n_chan_total, n_pols), dtype=bool)

    # 4. Fill the arrays by iterating through SPWs (much faster than querying)
    curr_chan = 0
    for spw in unique_spws:
        n_chan_spw = len(all_freqs_list[spw])

        # Mask for this SPW
        spw_mask = spw_ids == spw

        for ant in unique_ants:
            # Mask for this Antenna within this SPW
            idx = np.where(spw_mask & (ant_ids == ant))[0]

            if len(idx) > 0:
                # Map the slice into our pre-allocated array
                # Note: This handles time integrations if present
                final_data[ant, : len(idx), curr_chan : curr_chan + n_chan_spw, :] = (
                    main_data[idx]
                )
                final_flags[ant, : len(idx), curr_chan : curr_chan + n_chan_spw, :] = (
                    main_flags[idx]
                )

        curr_chan += n_chan_spw

    if isinstance(example_data, str):
        uv = pyuvdata.UVData()
        uv.read(example_data)
    else:
        uv = example_data

    uvcal = pyuvdata.UVCal()
    uvcal.Nants_data = len(ant_names)
    uvcal.Nfreqs = len(frequencies)
    uvcal.Njones = 2
    uvcal.Nspws = 1
    uvcal.Ntimes = 1
    uvcal.ant_array = [
        uv.telescope.antenna_numbers[
            np.where(np.array(uv.telescope.antenna_names) == name)[0][0]
        ]
        for name in ant_names
    ]
    uvcal.cal_style = "sky"
    uvcal.cal_type = "gain"
    uvcal.channel_width = np.full(uvcal.Nfreqs, frequencies[1] - frequencies[0])
    uvcal.flag_array = np.transpose(final_flags, (0, 2, 1, 3))
    uvcal.flex_spw_id_array = np.full(uvcal.Nfreqs, 0, dtype=int)
    uvcal.freq_array = frequencies
    uvcal.gain_convention = "multiply"
    uvcal.history = table_path
    uvcal.integration_time = np.array([10.0])
    uvcal.jones_array = np.array([-5, -6])
    uvcal.spw_array = np.array([0])
    uvcal.telescope = uv.telescope
    uvcal.wide_band = False
    uvcal.gain_array = np.transpose(final_data, (0, 2, 1, 3))
    uvcal.ref_antenna_name = ant_names[0]  # Dummy reference antenna
    uvcal.sky_catalog = "pipeline"
    uvcal.time_array = np.array([jd])
    uvcal.lst_array = pyuvdata.utils.get_lst_for_time(
        jd_array=uvcal.time_array, telescope_loc=uv.telescope.location
    )

    if not uvcal.check():
        print("WARNING: Check failed.")

    return uvcal


def read_caltable_safely(path, tmp_directory="/fast/rbyrne/caltable_temp_dir"):
    """
    This function replaces pyuvdata's UVCal.read() function due to a bug in that function.
    See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/1648.
    Deprecated function with resolution of that issue!
    """

    if os.path.isfile(path):  # Path is not an ms caltable
        cal = pyuvdata.UVCal()
        cal.read(path)
        return cal

    tb = tbl.table(path, readonly=True)
    if "SPECTRAL_WINDOW_ID" in tb.colnames():
        spw_col = tb.getcol("SPECTRAL_WINDOW_ID")
        unique_spws = np.unique(spw_col)
        unique_spws.sort()

        tb_spw = tbl.table(f"{path}/SPECTRAL_WINDOW", readonly=True)
        chan_freq = tb_spw.getcol("CHAN_FREQ")
        tb_spw.close()
    else:  # Only one spw
        tb.close()
        cal = pyuvdata.UVCal()
        cal.read(path)
        return cal

    for spw in unique_spws:
        subtable = tb.query(f"SPECTRAL_WINDOW_ID == {spw}")
        subtable.copy(f"{tmp_directory}/temp_spw{spw}.B", deep=True)
        subtable.close()

    tb.close()

    cal_objs = []
    time_array = []
    lst_array = []
    for spw_ind, spw in enumerate(unique_spws):
        cal = pyuvdata.UVCal()
        cal.read(f"{tmp_directory}/temp_spw{spw}.B")
        cal.select(frequencies=chan_freq[spw_ind, :])
        cal_objs.append(cal)
        time_array.append(np.mean(cal.time_array))
        lst_array.append(np.mean(cal.lst_array))

    for cal_ind, cal in enumerate(cal_objs):
        cal.time_array = np.array([np.mean(time_array)])
        cal.lst_array = np.array([np.mean(lst_array)])
        if cal_ind == 0:
            cal_concat = cal
        else:
            cal_concat += cal  # Do not use fast_concat method here

    cal_concat.spw_array = np.array([0])
    cal_concat.flex_spw_id_array = np.zeros_like(cal_concat.flex_spw_id_array)
    cal_concat.Nspws = 1

    # Delete temporary files
    # for spw in unique_spws:
    #    shutil.rmtree(f"{tmp_directory}/temp_spw{spw}.B")

    return cal_concat