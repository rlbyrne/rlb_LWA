#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
sys.path.append('/Users/ruby/Astro/pyuvdata')
import pyuvdata


def get_test_data():

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
    if data_path != model_path or model_use_model != data_use_model:
        data = pyuvdata.UVData()
        print('Reading data...')
        data.read_fhd(data_filelist, use_model=data_use_model)
        print('Done.')
    else:
        print('Using model for data')
        data = model.copy()

    # For testing, use one time only
    use_time = data.time_array[200000]
    data.select(times=use_time)
    model.select(times=use_time)

    # Need to check that the baseline ordering agrees between model and data

    return data, model


def calc_negloglikelihood_sky_cal(
    gains, data_visibilities, model_visibilities,
    gains_exp_mat_1, gains_exp_mat_2,
    data_stddev
):

    gains_expanded = (
        np.matmul(gains_exp_mat_1, gains)
        * np.matmul(gains_exp_mat_2, np.conj(gains))
    )
    prob = np.sum(np.abs(
        data_visibilities - gains_expanded*model_visibilities
    )**2)

    return (prob/data_stddev**2.)


def cost_function_sky_cal(
    x,
    N_ants,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    data_visibilities,
    data_stddev
):

    gains = x[:N_ants]+1j*x[N_ants:2*N_ants]

    cost = calc_negloglikelihood_sky_cal(
        gains, data_visibilities, model_visibilities,
        gains_exp_mat_1, gains_exp_mat_2,
        data_stddev
    )
    return cost


def calc_gains_grad(
    gains, model_visibilities, data_visibilities,
    gains_exp_mat_1, gains_exp_mat_2,
    data_stddev
):

    gains1_expanded = np.matmul(gains_exp_mat_1, gains)
    gains2_expanded = np.matmul(gains_exp_mat_2, gains)

    gains_grad_term_1 = (
        np.abs(gains1_expanded*model_visibilities)**2.*gains2_expanded
        - np.conj(data_visibilities)*gains1_expanded*model_visibilities
    )
    gains_grad_term_2 = (
        np.abs(np.conj(gains2_expanded)*model_visibilities)**2.*gains1_expanded
        - data_visibilities*gains2_expanded*np.conj(model_visibilities)
    )
    gains_grad = (2./data_stddev**2.)*(
        np.matmul(gains_exp_mat_2.T, gains_grad_term_1)
        + np.matmul(gains_exp_mat_1.T, gains_grad_term_2)
    )

    return gains_grad


def jac_function_sky_cal(
    x,
    N_ants,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    data_visibilities,
    data_stddev
):

    gains = x[:N_ants]+1j*x[N_ants:2*N_ants]

    gains_grad = calc_gains_grad(
        gains, model_visibilities, data_visibilities,
        gains_exp_mat_1, gains_exp_mat_2,
        data_stddev
    )

    grads = np.zeros(N_ants*2)
    grads[:N_ants] = np.real(gains_grad)
    grads[N_ants:2*N_ants] = np.imag(gains_grad)
    return grads


def calc_calibration_components(data):

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((data.Nbls, data.Nants_data), dtype=np.int)
    gains_exp_mat_2 = np.zeros((data.Nbls, data.Nants_data), dtype=np.int)
    for baseline in range(data.Nbls):
        gains_exp_mat_1[baseline, data.ant_1_array[baseline]] = 1
        gains_exp_mat_2[baseline, data.ant_2_array[baseline]] = 1

    return gains_exp_mat_1, gains_exp_mat_2,


def optimize_sky_cal(
    data_visibilities, model_visibilities,
    gains_exp_mat_1, gains_exp_mat_2,
    N_ants, data_stddev,
    gains_init=None, quiet=False
):

    method = 'CG'
    maxiter = 100000

    if gains_init is None:  # Initialize the gains to 1
        gains_init = np.full(N_ants, 1.+0.j)
    # Expand the initialized values
    x0 = np.concatenate((
        np.real(gains_init), np.imag(gains_init)
    ))

    # Minimize the cost function
    result = scipy.optimize.minimize(
        cost_function_sky_cal, x0, jac=jac_function_sky_cal,
        args=(
            N_ants, model_visibilities,
            gains_exp_mat_1, gains_exp_mat_2,
            data_visibilities, data_stddev
        ),
        method=method, options={'disp': True, 'maxiter': maxiter}
    )
    if not quiet:
        print(result.message)

    gains_fit = result.x[:N_ants]+1j*result.x[N_ants:2*N_ants]
    # Ensure that the angle of the gains is mean-zero
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(gains_fit))),
        np.mean(np.cos(np.angle(gains_fit)))
    )
    gains_fit *= np.cos(avg_angle) - 1j*np.sin(avg_angle)

    return gains_fit


def calibrate():

    data, model = get_test_data()
    gains_exp_mat_1, gains_exp_mat_2 = calc_calibration_components(data)

    data_stddev = 1.

    optimize_function_name = optimize_sky_cal
    gains_fit = optimize_function_name(
        data.data_array, model.data_array,
        gains_exp_mat_1, gains_exp_mat_2,
        data.Nants_data, data_stddev, gains_init
    )


def get_calibration_reference():

    cal_file_path = '/Users/ruby/Astro/fhd_rlb_perfreq_cal_Feb2021/calibration/1061316296_cal.sav'
    obs_file_path = '/Users/ruby/Astro/fhd_rlb_perfreq_cal_Feb2021/metadata/1061316296_obs.sav'
    cal_obj = pyuvdata.UVCal()
    cal_obj.read_fhd_cal(cal_file_path, obs_file_path)

    # Change parameters
    cal_obj.Nsources = None
    cal_obj.ref_antenna_name = None
    old_gains = np.copy(cal_obj.gain_array)
    cal_obj.gain_array = None
    cal_obj.baseline_range = [0., cal_obj.baseline_range[1]]
    # Clear flags
    cal_obj.flag_array = None


if __name__=='__main__':
    get_calibration_reference()
