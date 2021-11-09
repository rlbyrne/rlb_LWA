#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import scipy.optimize
import time

sys.path.append("/Users/ruby/Astro/pyuvdata")
import pyuvdata


def get_test_data(use_autos=False):

    model_path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021"
    model_use_model = True
    data_path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021"
    data_use_model = True
    obsid = "1061316296"
    pol = "XX"

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
    print("Reading model...")
    model.read_fhd(model_filelist, use_model=model_use_model)

    # For testing, use one time and a few frequencies only
    all_times = np.unique(model.time_array)
    use_times = all_times[int(model.Ntimes/2):int(model.Ntimes/2+3)]
    use_frequencies = model.freq_array[0, 100:110]
    model.select(times=use_times, frequencies=use_frequencies)

    if not use_autos:  # Remove autocorrelations
        bl_lengths = np.sqrt(np.sum(model.uvw_array ** 2.0, axis=1))
        non_autos = np.where(bl_lengths > 0.01)[0]
        model.select(blt_inds=non_autos)

    if data_path != model_path or model_use_model != data_use_model:
        data = pyuvdata.UVData()
        print("Reading data...")
        data.read_fhd(data_filelist, use_model=data_use_model)
        print("Done.")
        data.select(times=use_times, frequencies=use_frequencies)
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


def initialize_cal(data):

    cal = pyuvdata.UVCal()
    cal.Nants_data = data.Nants_data
    cal.Nants_telescope = data.Nants_telescope
    cal.Nfreqs = data.Nfreqs
    cal.Njones = 1
    cal.Nspws = 1
    cal.Ntimes = 1
    cal.ant_array = np.unique(np.concatenate((data.ant_1_array, data.ant_2_array)))
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
    cal.x_orientation = "east"
    cal.gain_array = np.full(
        (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
        1,
        dtype=complex,
    )
    cal.flag_array = np.full(
        (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
        False,
        dtype=bool,
    )
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
        axis=0
    )
    term1 = reformat_baselines_to_antenna_matrix(
        term1, gains_exp_mat_1, gains_exp_mat_2
    )
    term1 = np.transpose(term1, (1, 0, 2, 3))

    term2 = np.sum(
        gains2_times_conj_model[:, :, np.newaxis, :]
        * gains1_times_model[:, :, :, np.newaxis]
        * cov_mat[np.newaxis, :, :, :],
        axis=0
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
        hess[:, :, freq, freq, 1] += np.imag(terms3and4[:, :, freq])
        hess[:, :, freq, freq, 2] += np.real(terms3and4[:, :, freq])

    hess_reformatted = np.zeros((2, 2, Nants * Nfreqs, Nants * Nfreqs), dtype=float)
    hess_reformatted[0, 0, :, :] = hess[:, :, :, :, 0].reshape(
        Nants * Nfreqs, Nants * Nfreqs
    )
    hess_reformatted[0, 1, :, :] = hess[:, :, :, :, 1].reshape(
        Nants * Nfreqs, Nants * Nfreqs
    )
    hess_reformatted[1, 0, :, :] = (
        hess[:, :, :, :, 1].reshape(Nants * Nfreqs, Nants * Nfreqs)
    ).T
    hess_reformatted[1, 1, :, :] = hess[:, :, :, :, 2].reshape(
        Nants * Nfreqs, Nants * Nfreqs
    )
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


def calibrate():

    data, model = get_test_data(use_autos=False)

    cal = initialize_cal(data)

    # Format visibilities
    data_visibilities = np.zeros((data.Ntimes, data.Nbls, data.Nfreqs), dtype=complex)
    model_visibilities = np.zeros((data.Ntimes, data.Nbls, data.Nfreqs), dtype=complex)
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
    gains_exp_mat_1 = np.zeros((metadata_reference.Nbls, metadata_reference.Nants_data), dtype=int)
    gains_exp_mat_2 = np.zeros((metadata_reference.Nbls, metadata_reference.Nants_data), dtype=int)
    antenna_list = np.unique([metadata_reference.ant_1_array, metadata_reference.ant_2_array])
    for baseline in range(metadata_reference.Nbls):
        gains_exp_mat_1[
            baseline, np.where(antenna_list == metadata_reference.ant_1_array[baseline])
        ] = 1
        gains_exp_mat_2[
            baseline, np.where(antenna_list == metadata_reference.ant_2_array[baseline])
        ] = 1

    # method = 'CG'
    method = "Newton-CG"
    xtol = 1e-10  # Defaults to 1e-05

    # Initialize gains
    gain_init_noise = 0.01
    gains_init = np.random.normal(
        1.0, gain_init_noise, size=(cal.Nants_data, cal.Nfreqs)
    ) + 1.0j * np.random.normal(0.0, gain_init_noise, size=(cal.Nants_data, cal.Nfreqs))
    # Expand the initialized values
    x0 = np.stack((np.real(gains_init), np.imag(gains_init)), axis=0).flatten()

    # Define covariance matrix
    cov_mat = np.identity(cal.Nfreqs)
    cov_mat = np.repeat(cov_mat[np.newaxis, :, :], metadata_reference.Nbls, axis=0)
    cov_mat = cov_mat.reshape((metadata_reference.Nbls, metadata_reference.Nfreqs, metadata_reference.Nfreqs))

    # Minimize the cost function
    print("Beginning optimization")
    result = scipy.optimize.minimize(
        cost_function_dw_cal,
        x0,
        args=(
            cal.Nants_data,
            cal.Nfreqs,
            metadata_reference.Nbls,
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

    gains_fit = np.reshape(result.x, (2, cal.Nants_data, cal.Nfreqs))
    gains_fit = (
        gains_fit[0,] + 1.0j * gains_fit[1,]
    )
    # Ensure that the angle of the gains is mean-zero for each frequency
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(gains_fit)), axis=0),
        np.mean(np.cos(np.angle(gains_fit)), axis=0),
    )
    gains_fit *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    print(np.min(np.real(gains_fit)))
    print(np.max(np.real(gains_fit)))
    print(np.min(np.imag(gains_fit)))
    print(np.max(np.imag(gains_fit)))
    print(np.mean(np.real(gains_fit)))
    print(np.mean(np.imag(gains_fit)))
    print("Nfreqs: {}".format(cal.Nfreqs))


def get_calibration_reference():

    cal_file_path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_perfreq_cal_Feb2021/calibration/1061316296_cal.sav"
    obs_file_path = "/Users/ruby/Astro/FHD_outputs/fhd_rlb_perfreq_cal_Feb2021/metadata/1061316296_obs.sav"
    cal_obj = pyuvdata.UVCal()
    cal_obj.read_fhd_cal(cal_file_path, obs_file_path)
    gains = np.copy(cal_obj.gain_array)


if __name__ == "__main__":
    start = time.time()
    calibrate()
    end = time.time()
    print(f"Runtime {(end - start)/60.} minutes.")
