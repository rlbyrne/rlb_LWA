#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import scipy.optimize
sys.path.append('/Users/ruby/Astro/pyuvdata')
import pyuvdata
import time


def get_test_data(use_autos=True):

    model_path = '/Users/ruby/Astro/fhd_rlb_model_GLEAM_Aug2021'
    model_use_model = True
    data_path = '/Users/ruby/Astro/fhd_rlb_model_GLEAM_Aug2021'
    data_use_model = True
    obsid = '1061316296'
    pol = 'XX'

    model_filelist = ['{}/{}'.format(model_path, file) for file in [
        'vis_data/{}_vis_{}.sav'.format(obsid, pol),
        'vis_data/{}_vis_model_{}.sav'.format(obsid, pol),
        'vis_data/{}_flags.sav'.format(obsid),
        'metadata/{}_params.sav'.format(obsid),
        'metadata/{}_settings.txt'.format(obsid),
        'metadata/{}_layout.sav'.format(obsid)
    ]]
    data_filelist = ['{}/{}'.format(data_path, file) for file in [
        'vis_data/{}_vis_{}.sav'.format(obsid, pol),
        'vis_data/{}_vis_model_{}.sav'.format(obsid, pol),
        'vis_data/{}_flags.sav'.format(obsid),
        'metadata/{}_params.sav'.format(obsid),
        'metadata/{}_settings.txt'.format(obsid),
        'metadata/{}_layout.sav'.format(obsid)
    ]]

    model = pyuvdata.UVData()
    print('Reading model...')
    model.read_fhd(model_filelist, use_model=model_use_model)

    # For testing, use one time and a few frequencies only
    use_time = model.time_array[200000]
    use_frequencies = model.freq_array[0, 100:200]
    model.select(times=use_time, frequencies=use_frequencies)

    if not use_autos:  # Remove autocorrelations
        bl_lengths = np.sqrt(np.sum(model.uvw_array**2., axis=1))
        non_autos = np.where(bl_lengths > 0.01)[0]
        model.select(blt_inds=non_autos)

    if data_path != model_path or model_use_model != data_use_model:
        data = pyuvdata.UVData()
        print('Reading data...')
        data.read_fhd(data_filelist, use_model=data_use_model)
        print('Done.')
        data.select(times=use_time, frequencies=use_frequencies)
        if not use_autos:  # Remove autocorrelations
            bl_lengths = np.sqrt(np.sum(data.uvw_array**2., axis=1))
            non_autos = np.where(bl_lengths > 0.01)[0]
            data.select(blt_inds=non_autos)
    else:
        print('Using model for data')
        data = model.copy()

    # For testing, use one time only
    use_time = data.time_array[200000]
    use_frequencies = data.freq_array[0, 100]
    data.select(times=use_time, frequencies=use_frequencies)
    model.select(times=use_time, frequencies=use_frequencies)

    # Need to check that the baseline ordering agrees between model and data

    return data, model


def initialize_cal(data):

    cal = pyuvdata.UVCal()
    cal.Nants_data = data.Nants_data
    cal.Nants_telescope = data.Nants_telescope
    cal.Nfreqs = data.Nfreqs
    cal.Njones = 1
    cal.Nspws = 1
    cal.Ntimes = 1
    cal.ant_array = np.intersect1d(data.ant_1_array, data.ant_2_array)
    cal.antenna_names = data.antenna_names
    cal.antenna_numbers = data.antenna_numbers
    cal.cal_style = 'sky'
    cal.cal_type = 'gain'
    cal.channel_width = data.channel_width
    cal.freq_array = data.freq_array
    cal.gain_convention = 'divide'
    cal.history = ''
    cal.integration_time = np.mean(data.integration_time)
    cal.jones_array = np.array([-5])
    cal.spw_array = data.spw_array
    cal.telescope_name = data.telescope_name
    cal.time_array = np.array([np.mean(data.time_array)])
    cal.x_orientation = 'east'
    cal.gain_array = np.full((
        cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones
    ), 1, dtype=complex)
    cal.flag_array = np.full((
        cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones
    ), False, dtype=bool)
    cal.quality_array = np.full((
        cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones
    ), 1., dtype=float)
    cal.ref_antenna_name = ''
    cal.sky_catalog = 'GLEAM_bright_sources'
    cal.sky_field = 'phase center (RA, Dec): ({}, {})'.format(
        np.degrees(np.mean(data.phase_center_app_ra)),
        np.degrees(np.mean(data.phase_center_app_dec))
    )

    if not cal.check():
        print('ERROR: UVCal check failed.')
        sys.exit(1)

    return cal


def cost_function_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nblts,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities
):

    gains = np.reshape(x, (2, Nants, Nfreqs,))
    gains = gains[0, ] + 1.j*gains[1, ]

    print(f'model_vis: {np.shape(model_visibilities)}')
    print(f'data_vis: {np.shape(data_visibilities)}')
    print(f'gain_exp_mat1: {np.shape(gains_exp_mat_1)}')
    print(f'gain_exp_mat2: {np.shape(gains_exp_mat_2)}')
    print(f'cov_mat: {np.shape(cov_mat)}')
    print(f'gains: {np.shape(gains)}')

    gains_expanded = (
        np.matmul(gains_exp_mat_1, gains)
        * np.matmul(gains_exp_mat_2, np.conj(gains))
    )
    res_vec = data_visibilities - gains_expanded*model_visibilities
    weighted_part2 = np.squeeze(np.matmul(res_vec[:, np.newaxis, :], cov_mat))
    cost = np.real(np.sum(np.conj(res_vec)*weighted_part2))

    print('Cost func. eval.')

    return cost


def grad_function_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nblts,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities
):

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = gains[0, ] + 1.j*gains[1, ]

    gains1_expanded = np.matmul(gains_exp_mat_1, gains)
    gains2_expanded = np.matmul(gains_exp_mat_2, gains)
    term1_part1 = gains1_expanded*model_visibilities
    term2_part1 = gains2_expanded*np.conj(model_visibilities)
    cost_term = data_visibilities - gains1_expanded*np.conj(gains2_expanded)*model_visibilities
    weighted_part2 = np.squeeze(np.matmul(cost_term[:, np.newaxis, :], cov_mat))
    term1 = np.matmul(gains_exp_mat_2.T, term1_part1*np.conj(weighted_part2))
    term2 = np.matmul(gains_exp_mat_1.T, term2_part1*weighted_part2)
    grad = -2*(term1 + term2)

    grad = np.stack((np.real(grad), np.imag(grad)), axis=0).flatten()

    return grad


def cost_function_sky_cal(
    x,
    Nants,
    Nfreqs,
    Nblts,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities
):

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = gains[0, ] + 1.j*gains[1, ]

    gains_expanded = (
        np.matmul(gains_exp_mat_1, gains)
        * np.matmul(gains_exp_mat_2, np.conj(gains))
    )
    res_vec = data_visibilities - gains_expanded*model_visibilities

    cost = np.sum(np.abs(res_vec)**2)

    return cost


def calibrate():

    data, model = get_test_data()

    cal = initialize_cal(data)

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((data.Nblts, cal.Nants_data), dtype=int)
    gains_exp_mat_2 = np.zeros((data.Nblts, cal.Nants_data), dtype=int)
    antenna_list = np.unique([data.ant_1_array, data.ant_2_array])
    for baseline in range(data.Nblts):
        gains_exp_mat_1[
            baseline, np.where(antenna_list == data.ant_1_array[baseline])
        ] = 1
        gains_exp_mat_2[
            baseline, np.where(antenna_list == data.ant_2_array[baseline])
        ] = 1

    #method = 'CG'
    method = 'Newton-CG'
    #maxiter = 100000

    # Initialize gains
    gain_init_noise = .001
    gains_init = (np.random.normal(
        1., gain_init_noise, size=(cal.Nants_data, cal.Nfreqs)
    ) + 1.j*np.random.normal(
        0., gain_init_noise, size=(cal.Nants_data, cal.Nfreqs)
    ))
    # Expand the initialized values
    x0 = np.stack((
        np.real(gains_init), np.imag(gains_init)
    ), axis=0).flatten()

    # Define covariance matrix
    cov_mat = np.identity(cal.Nfreqs)
    cov_mat = np.repeat(cov_mat[np.newaxis, :, :], data.Nblts, axis=0)
    cov_mat.reshape((cal.Nfreqs, cal.Nfreqs, data.Nblts))

    # Minimize the cost function
    print('Beginning optimization')
    result = scipy.optimize.minimize(
        cost_function_dw_cal, x0, jac=grad_function_dw_cal,
        args=(
            cal.Nants_data,
            cal.Nfreqs,
            data.Nblts,
            np.squeeze(model.data_array, axis=(1, 3)),
            gains_exp_mat_1,
            gains_exp_mat_2,
            cov_mat,
            np.squeeze(data.data_array, axis=(1, 3))
        ),
        options={'disp': True}
    )
    print(result.message)

    gains_fit = np.reshape(result.x, (2, cal.Nants_data, cal.Nfreqs))
    gains_fit = gains_fit[0, ] + 1.j*gains_fit[1, ]
    # Ensure that the angle of the gains is mean-zero for each frequency
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(gains_fit)), axis=0),
        np.mean(np.cos(np.angle(gains_fit)), axis=0)
    )
    gains_fit *= np.cos(avg_angle) - 1j*np.sin(avg_angle)

    print(np.min(np.real(gains_fit)))
    print(np.max(np.real(gains_fit)))
    print(np.min(np.imag(gains_fit)))
    print(np.max(np.imag(gains_fit)))
    print(np.mean(np.real(gains_fit)))
    print(np.mean(np.imag(gains_fit)))
    print('Nfreqs: {}'.format(cal.Nfreqs))


def get_calibration_reference():

    cal_file_path = '/Users/ruby/Astro/fhd_rlb_perfreq_cal_Feb2021/calibration/1061316296_cal.sav'
    obs_file_path = '/Users/ruby/Astro/fhd_rlb_perfreq_cal_Feb2021/metadata/1061316296_obs.sav'
    cal_obj = pyuvdata.UVCal()
    cal_obj.read_fhd_cal(cal_file_path, obs_file_path)
    gains = np.copy(cal_obj.gain_array)


if __name__=='__main__':
    start = time.time()
    calibrate()
    end = time.time()
    print(f"Runtime {(end - start)/60.} minutes.")
