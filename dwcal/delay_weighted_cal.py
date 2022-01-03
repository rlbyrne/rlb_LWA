#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import scipy.optimize
import time
import pyuvdata


def get_test_data(
    model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    model_use_model=True,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    use_autos=False,
    debug_limit_freqs=None,
):

    model_filelist = [
        "{}/{}".format(model_path, file)
        for file in [
            "vis_data/{}_vis_{}.sav".format(obsid, pol),
            "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
            "vis_data/{}_flags.sav".format(obsid),
            "metadata/{}_params.sav".format(obsid),
            "metadata/{}_settings.txt".format(obsid),
            "metadata/{}_layout.sav".format(obsid),
        ]
    ]
    data_filelist = [
        "{}/{}".format(data_path, file)
        for file in [
            "vis_data/{}_vis_{}.sav".format(obsid, pol),
            "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
            "vis_data/{}_flags.sav".format(obsid),
            "metadata/{}_params.sav".format(obsid),
            "metadata/{}_settings.txt".format(obsid),
            "metadata/{}_layout.sav".format(obsid),
        ]
    ]

    model = pyuvdata.UVData()
    if model_use_model:
        print(
            f"Loading model from {model_path}, using the FHD run's model visibilities."
        )
    else:
        print(
            f"Loading model from {model_path}, using the FHD run's data visibilities."
        )
    model.read_fhd(model_filelist, use_model=model_use_model)

    # Average across time
    model.downsample_in_time(n_times_to_avg=model.Ntimes)

    if debug_limit_freqs is not None:  # Limit frequency axis for debugging
        min_freq_channel = round(model.Nfreqs / 2 - debug_limit_freqs / 2)
        use_frequencies = model.freq_array[
            0, min_freq_channel : round(min_freq_channel + debug_limit_freqs)
        ]
        model.select(frequencies=use_frequencies)

    if not use_autos:  # Remove autocorrelations
        bl_lengths = np.sqrt(np.sum(model.uvw_array ** 2.0, axis=1))
        non_autos = np.where(bl_lengths > 0.01)[0]
        model.select(blt_inds=non_autos)

    if data_path != model_path or model_use_model != data_use_model:
        data = pyuvdata.UVData()
        if data_use_model:
            print(
                f"Loading data from {data_path}, using the FHD run's model visibilities."
            )
        else:
            print(
                f"Loading data from {data_path}, using the FHD run's data visibilities."
            )
        data.read_fhd(data_filelist, use_model=data_use_model)
        data.downsample_in_time(n_times_to_avg=data.Ntimes)
        if debug_limit_freqs is not None:
            data.select(frequencies=use_frequencies)
        if not use_autos:  # Remove autocorrelations
            bl_lengths = np.sqrt(np.sum(data.uvw_array ** 2.0, axis=1))
            non_autos = np.where(bl_lengths > 0.01)[0]
            data.select(blt_inds=non_autos)
    else:
        print("Using model for data")
        data = model.copy()

    # Ensure ordering matches between the data and model
    if np.max(np.abs(data.baseline_array - model.baseline_array)) > 0.0:
        data.reorder_blts()
        model.reorder_blts()
    if np.max(np.abs(data.freq_array - model.freq_array)) > 0.0:
        data.reorder_freqs(channel_order="freq")
        model.reorder_freqs(channel_order="freq")

    return data, model


def initialize_cal(data, antenna_list, gains=None):

    cal = pyuvdata.UVCal()
    cal.Nants_data = data.Nants_data
    cal.Nants_telescope = data.Nants_telescope
    cal.Nfreqs = data.Nfreqs
    cal.Njones = 1
    cal.Nspws = 1
    cal.Ntimes = 1
    cal.ant_array = antenna_list
    cal.antenna_names = data.antenna_names
    cal.antenna_numbers = data.antenna_numbers
    cal.cal_style = "sky"
    cal.cal_type = "gain"
    cal.channel_width = data.channel_width
    cal.freq_array = data.freq_array
    cal.gain_convention = "divide"
    cal.history = ""
    cal.integration_time = np.mean(data.integration_time)
    cal.jones_array = np.array([-5])
    cal.spw_array = data.spw_array
    cal.telescope_name = data.telescope_name
    cal.time_array = np.array([np.mean(data.time_array)])
    cal.time_range = np.array([np.min(data.time_array), np.max(data.time_array)])
    cal.x_orientation = "east"
    if gains is None:  # Set all gains to 1
        cal.gain_array = np.full(
            (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
            1,
            dtype=complex,
        )
    else:
        cal.gain_array = gains[:, np.newaxis, :, np.newaxis, np.newaxis]
    cal.flag_array = np.full(
        (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
        False,
        dtype=bool,
    )  # Does not yet support flags
    cal.quality_array = np.full(
        (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
        1.0,
        dtype=float,
    )
    cal.ref_antenna_name = ""
    cal.sky_catalog = "GLEAM_bright_sources"
    cal.sky_field = "phase center (RA, Dec): ({}, {})".format(
        np.degrees(np.mean(data.phase_center_app_ra)),
        np.degrees(np.mean(data.phase_center_app_dec)),
    )

    if not cal.check():
        print("ERROR: UVCal check failed.")
        sys.exit(1)

    return cal


def cost_function_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    gains = np.reshape(x, (2, Nants, Nfreqs,))
    gains = (
        gains[0,] + 1.0j * gains[1,]
    )

    gains_expanded = np.matmul(gains_exp_mat_1, gains) * np.matmul(
        gains_exp_mat_2, np.conj(gains)
    )
    res_vec = data_visibilities - gains_expanded[np.newaxis, :, :] * model_visibilities
    weighted_part2 = np.squeeze(np.matmul(res_vec[:, :, np.newaxis, :], cov_mat))
    cost = np.real(np.sum(np.conj(np.squeeze(res_vec)) * weighted_part2))

    print("Cost func. eval.")

    return cost


def jac_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = (
        gains[0,] + 1.0j * gains[1,]
    )

    gains1_expanded = np.matmul(gains_exp_mat_1, gains)
    gains2_expanded = np.matmul(gains_exp_mat_2, gains)
    term1_part1 = gains1_expanded[np.newaxis, :, :] * model_visibilities
    term2_part1 = gains2_expanded[np.newaxis, :, :] * np.conj(model_visibilities)
    cost_term = (
        data_visibilities
        - gains1_expanded * np.conj(gains2_expanded) * model_visibilities
    )
    weighted_part2 = np.squeeze(np.matmul(cost_term[:, :, np.newaxis, :], cov_mat))
    term1 = np.sum(
        np.matmul(gains_exp_mat_2.T, term1_part1 * np.conj(weighted_part2)), axis=0
    )
    term2 = np.sum(np.matmul(gains_exp_mat_1.T, term2_part1 * weighted_part2), axis=0)
    grad = -2 * (term1 + term2)

    grad = np.stack((np.real(grad), np.imag(grad)), axis=0).flatten()

    return grad


def reformat_baselines_to_antenna_matrix(bl_array, gains_exp_mat_1, gains_exp_mat_2):
    # Reformat an array indexed in baselines into a matrix with antenna indices

    (Nbls, Nants) = np.shape(gains_exp_mat_1)
    antenna_matrix = np.zeros_like(bl_array[0,], dtype=bl_array.dtype)
    antenna_matrix = np.repeat(
        np.repeat(antenna_matrix[np.newaxis,], Nants, axis=0)[np.newaxis,],
        Nants,
        axis=0,
    )
    antenna_numbers = np.arange(Nants)
    antenna1_num = np.matmul(gains_exp_mat_1, antenna_numbers)
    antenna2_num = np.matmul(gains_exp_mat_2, antenna_numbers)
    for bl_ind in range(Nbls):
        antenna_matrix[antenna1_num[bl_ind], antenna2_num[bl_ind],] = bl_array[
            bl_ind,
        ]
    return antenna_matrix


def hess_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = (
        gains[0,] + 1.0j * gains[1,]
    )

    gains1_expanded = np.matmul(gains_exp_mat_1, gains)
    gains2_expanded = np.matmul(gains_exp_mat_2, gains)

    gains1_times_model = gains1_expanded[np.newaxis, :, :] * model_visibilities
    gains2_times_conj_model = gains2_expanded[np.newaxis, :, :] * np.conj(
        model_visibilities
    )

    term1 = np.sum(
        gains1_times_model[:, :, np.newaxis, :]
        * gains2_times_conj_model[:, :, :, np.newaxis]
        * np.conj(cov_mat[np.newaxis, :, :, :]),
        axis=0,
    )
    term1 = reformat_baselines_to_antenna_matrix(
        term1, gains_exp_mat_1, gains_exp_mat_2
    )
    term1 = np.transpose(term1, (1, 0, 2, 3))

    term2 = np.sum(
        gains2_times_conj_model[:, :, np.newaxis, :]
        * gains1_times_model[:, :, :, np.newaxis]
        * cov_mat[np.newaxis, :, :, :],
        axis=0,
    )
    term2 = reformat_baselines_to_antenna_matrix(
        term2, gains_exp_mat_1, gains_exp_mat_2
    )
    terms1and2 = 2 * (term1 + term2)

    # hess elements are ant_c, ant_d, freq_f0, freq_f1, and real/imag pair
    # The real/imag pairs are in order [real-real, real-imag, and imag-imag]
    hess = np.zeros((Nants, Nants, Nfreqs, Nfreqs, 3), dtype=float)
    hess[:, :, :, :, 0] = np.real(terms1and2)
    hess[:, :, :, :, 1] = np.imag(terms1and2)
    hess[:, :, :, :, 2] = -np.real(terms1and2)

    term3 = np.sum(
        np.conj(model_visibilities)
        * np.sum(
            cov_mat[np.newaxis, :, :, :]
            * (
                data_visibilities
                - gains1_expanded[np.newaxis, :, :]
                * np.conj(gains2_expanded[np.newaxis, :, :])
                * model_visibilities
            )[:, :, :, np.newaxis],
            axis=3,
        ),
        axis=0,
    )
    term3 = reformat_baselines_to_antenna_matrix(
        term3, gains_exp_mat_1, gains_exp_mat_2
    )
    term4 = np.transpose(np.conj(term3), (1, 0, 2))
    terms3and4 = -2 * (term3 + term4)
    for freq in range(Nfreqs):
        hess[:, :, freq, freq, 0] += np.real(terms3and4[:, :, freq])
        hess[:, :, freq, freq, 1] -= np.imag(terms3and4[:, :, freq])
        hess[:, :, freq, freq, 2] += np.real(terms3and4[:, :, freq])

    hess_reformatted = np.zeros((2, Nants * Nfreqs, 2, Nants * Nfreqs), dtype=float)
    hess_reformatted[0, :, 0, :] = np.transpose(
        hess[:, :, :, :, 0], (0, 2, 1, 3)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    hess_reformatted[0, :, 1, :] = np.transpose(
        hess[:, :, :, :, 1], (0, 2, 1, 3)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    hess_reformatted[1, :, 0, :] = np.transpose(
        hess[:, :, :, :, 1], (1, 3, 0, 2)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    hess_reformatted[1, :, 1, :] = np.transpose(
        hess[:, :, :, :, 2], (0, 2, 1, 3)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    del hess
    hess_reformatted = hess_reformatted.reshape(2 * Nants * Nfreqs, 2 * Nants * Nfreqs)

    return hess_reformatted


def cost_function_sky_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = (
        gains[0,] + 1.0j * gains[1,]
    )

    gains_expanded = np.matmul(gains_exp_mat_1, gains) * np.matmul(
        gains_exp_mat_2, np.conj(gains)
    )
    res_vec = data_visibilities - gains_expanded[np.newaxis, :, :] * model_visibilities

    cost = np.sum(np.abs(res_vec) ** 2)

    return cost


def get_cov_mat_identity(Nfreqs, Nbls):

    cov_mat = np.identity(Nfreqs)
    cov_mat = np.repeat(cov_mat[np.newaxis, :, :], Nbls, axis=0)
    cov_mat = cov_mat.reshape((Nbls, Nfreqs, Nfreqs))
    return cov_mat


def get_cov_mat_no_wedge(
    Nfreqs, Nbls, uvw_array, freq_array, wedge_buffer_factor=1.2, min_baselines=100
):

    c = 3.0 * 10 ** 8  # Speed of light
    bl_lengths = np.sqrt(np.sum(uvw_array ** 2.0, axis=1))
    freq_step = (np.max(freq_array) - np.min(freq_array)) / (Nfreqs - 1)
    delay_step = 1 / (2 * (np.max(freq_array) - np.min(freq_array)))
    delay_array = np.arange(
        -(Nfreqs - 1) * delay_step, Nfreqs * delay_step, delay_step, dtype=float
    )
    delay_weighting = np.zeros((Nbls, 2 * Nfreqs - 1))
    for delay_ind, delay_val in enumerate(delay_array):
        window_bls = np.where(wedge_buffer_factor * bl_lengths / c < np.abs(delay_val))[
            0
        ]
        if len(window_bls) >= min_baselines:
            delay_weighting[window_bls, delay_ind] = 1
        else:
            # Use all baselines
            delay_weighting[:, delay_ind] = 1

    # Shift delay zero point to the start
    delay_weighting = np.fft.ifftshift(delay_weighting, axes=1)
    # Fourier transform
    freq_weighting = np.fft.fft(delay_weighting, axis=1)
    # Result is symmetric and real, so keep half the values and real part only
    freq_weighting = np.real(freq_weighting[:, 0:Nfreqs])

    cov_mat = np.zeros((Nbls, Nfreqs, Nfreqs), dtype=float)
    # Set off-diagonals
    for freq_diff in range(1, Nfreqs):
        for start_freq in range(Nfreqs - freq_diff):
            cov_mat[:, start_freq, start_freq + freq_diff] = freq_weighting[
                :, freq_diff
            ]
    cov_mat += np.transpose(cov_mat, axes=(0, 2, 1))
    # Set diagonals
    for freq_ind in range(Nfreqs):
        cov_mat[:, freq_ind, freq_ind] = freq_weighting[:, 0]
    return cov_mat


def apply_calibration(
    cal,
    calibrated_data_savefile,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    debug_limit_freqs=None,
):

    data_filelist = [
        "{}/{}".format(data_path, file)
        for file in [
            "vis_data/{}_vis_{}.sav".format(obsid, pol),
            "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
            "vis_data/{}_flags.sav".format(obsid),
            "metadata/{}_params.sav".format(obsid),
            "metadata/{}_settings.txt".format(obsid),
            "metadata/{}_layout.sav".format(obsid),
        ]
    ]

    data = pyuvdata.UVData()
    data.read_fhd(data_filelist, use_model=data_use_model)

    if debug_limit_freqs is not None:
        min_freq_channel = round(data.Nfreqs / 2 - debug_limit_freqs / 2)
        use_frequencies = data.freq_array[
            0, min_freq_channel : round(min_freq_channel + debug_limit_freqs)
        ]
        data.select(frequencies=use_frequencies)

    data_calibrated = pyuvdata.utils.uvcalibrate(
        data, cal, inplace=False, time_check=False
    )
    return data_calibrated


def calibrate(
    model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    model_use_model=True,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    use_autos=False,
    use_wedge_exclusion=False,
    cal_savefile=None,
    calibrated_data_savefile=None,
    log_file_path=None,
    debug_limit_freqs=None,  # Set to number of freq channels to use
):

    if log_file_path is not None:
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    start = time.time()

    start_read_data = time.time()
    data, model = get_test_data(
        model_path=model_path,
        model_use_model=model_use_model,
        data_path=data_path,
        data_use_model=data_use_model,
        obsid=obsid,
        pol=pol,
        use_autos=use_autos,
        debug_limit_freqs=debug_limit_freqs,
    )
    end_read_data = time.time()
    print(f"Time reading data: {(end_read_data - start_read_data)/60.} minutes")
    sys.stdout.flush()

    Nants = data.Nants_data
    Nbls = data.Nbls
    Ntimes = data.Ntimes
    Nfreqs = data.Nfreqs

    # Format visibilities
    data_visibilities = np.zeros((Ntimes, Nbls, Nfreqs), dtype=complex)
    model_visibilities = np.zeros((Ntimes, Nbls, Nfreqs), dtype=complex)
    for time_ind, time_val in enumerate(np.unique(data.time_array)):
        data_copy = data.copy()
        model_copy = model.copy()
        data_copy.select(times=time_val)
        model_copy.select(times=time_val)
        data_copy.reorder_blts()
        model_copy.reorder_blts()
        data_copy.reorder_freqs(channel_order="freq")
        model_copy.reorder_freqs(channel_order="freq")
        if time_ind == 0:
            metadata_reference = data_copy.copy(metadata_only=True)
        model_visibilities[time_ind, :, :] = np.squeeze(
            model_copy.data_array, axis=(1, 3)
        )
        data_visibilities[time_ind, :, :] = np.squeeze(
            data_copy.data_array, axis=(1, 3)
        )

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((Nbls, Nants), dtype=int)
    gains_exp_mat_2 = np.zeros((Nbls, Nants), dtype=int)
    antenna_list = np.unique(
        [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
    )
    for baseline in range(metadata_reference.Nbls):
        gains_exp_mat_1[
            baseline, np.where(antenna_list == metadata_reference.ant_1_array[baseline])
        ] = 1
        gains_exp_mat_2[
            baseline, np.where(antenna_list == metadata_reference.ant_2_array[baseline])
        ] = 1

    # method = 'CG'
    method = "Newton-CG"
    # xtol = 1e-10  # Defaults to 1e-05
    xtol = 1e-5

    # Initialize gains
    gain_init_noise = 0.001
    gains_init = np.random.normal(
        1.0, gain_init_noise, size=(Nants, Nfreqs),
    ) + 1.0j * np.random.normal(0.0, gain_init_noise, size=(Nants, Nfreqs),)
    # Expand the initialized values
    x0 = np.stack((np.real(gains_init), np.imag(gains_init)), axis=0).flatten()

    start_cov_mat = time.time()
    if use_wedge_exclusion:
        print(f"use_wedge_exclusion=True: Generating wedge excluding covariance matrix")
        sys.stdout.flush()
        cov_mat = get_cov_mat_no_wedge(
            Nfreqs, Nbls, metadata_reference.uvw_array, metadata_reference.freq_array
        )
    else:
        print(f"use_wedge_exclusion=False: Covariance matrix is the identity")
        sys.stdout.flush()
        cov_mat = get_cov_mat_identity(Nfreqs, Nbls)
    end_cov_mat = time.time()
    print(
        f"Time generating covariance matrix: {(end_cov_mat - start_cov_mat)/60.} minutes"
    )
    sys.stdout.flush()

    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_function_dw_cal,
        x0,
        args=(
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            cov_mat,
            data_visibilities,
        ),
        method=method,
        jac=jac_dw_cal,
        hess=hess_dw_cal,
        options={"disp": True, "xtol": xtol},
    )
    print(result.message)
    end_optimize = time.time()
    print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
    sys.stdout.flush()

    gains_fit = np.reshape(result.x, (2, Nants, Nfreqs))
    gains_fit = (
        gains_fit[0,] + 1.0j * gains_fit[1,]
    )
    # Ensure that the angle of the gains is mean-zero for each frequency
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(gains_fit)), axis=0),
        np.mean(np.cos(np.angle(gains_fit)), axis=0),
    )
    gains_fit *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Create cal object
    cal = initialize_cal(data, antenna_list, gains=gains_fit)
    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    # Apply calibration
    if calibrated_data_savefile is not None:
        calibrated_data = apply_calibration(
            cal,
            calibrated_data_savefile,
            data_path=data_path,
            data_use_model=data_use_model,
            obsid=obsid,
            pol=pol,
            debug_limit_freqs=debug_limit_freqs,
        )
        print(f"Saving calibrated data to {calibrated_data_savefile}")
        sys.stdout.flush()
        calibrated_data.write_uvfits(calibrated_data_savefile)

    end = time.time()
    print(f"Total runtime: {(end - start)/60.} minutes")

    if log_file_path is not None:
        sys.stdout.close()


def get_calibration_reference():

    cal_file_path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_perfreq_cal_Feb2021/calibration/1061316296_cal.sav"
    obs_file_path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_perfreq_cal_Feb2021/metadata/1061316296_obs.sav"
    cal_obj = pyuvdata.UVCal()
    cal_obj.read_fhd_cal(cal_file_path, obs_file_path)
    gains = np.copy(cal_obj.gain_array)


if __name__ == "__main__":
    start = time.time()
    calibrate(
        cal_savefile="/Users/ruby/Astro/test_calibration/test_cal_savefile.calfits",
        calibrated_data_savefile="/Users/ruby/Astro/test_calibration/test_cal_data.uvfits",
        log_file_path="/Users/ruby/Astro/test_calibration/test_cal_log.txt",
        debug_limit_freqs=10,
    )
    end = time.time()
    print(f"Runtime {(end - start)/60.} minutes.")
