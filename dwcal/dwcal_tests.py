import delay_weighted_cal as dwcal
import numpy as np


def test_grad_real(
    test_ant,
    test_freq,
    delta_gains,
    gains_init,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    print("*******")
    print("Testing the gradient calculation, real part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant*Nfreqs+test_freq

    negloglikelihood0 = dwcal.cost_function_dw_cal(
        gains0_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )
    negloglikelihood1 = dwcal.cost_function_dw_cal(
        gains1_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )
    grad = dwcal.jac_dw_cal(
        gains_init_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    print(f"Empirical value: {(negloglikelihood1-negloglikelihood0)/delta_gains}")
    print(f"Calculated value: {grad[test_ind]}")


def test_grad_imag(
    test_ant,
    test_freq,
    delta_gains,
    gains_init,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    print("*******")
    print("Testing the gradient calculation, imaginary part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= 1j * delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += 1j * delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = Nfreqs*Nants+test_ant*Nfreqs+test_freq

    negloglikelihood0 = dwcal.cost_function_dw_cal(
        gains0_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )
    negloglikelihood1 = dwcal.cost_function_dw_cal(
        gains1_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )
    grad = dwcal.jac_dw_cal(
        gains_init_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    print(f"Empirical value: {(negloglikelihood1-negloglikelihood0)/delta_gains}")
    print(f"Calculated value: {grad[test_ind]}")


def test_hess_real_real(
    test_ant,
    test_freq,
    readout_ant,
    readout_freq,
    delta_gains,
    gains_init,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    cov_mat,
    data_visibilities,
):

    print("*******")
    print("Testing the hessian calculation, real-real part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= 1j * delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += 1j * delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant*Nfreqs+test_freq
    readout_ind = readout_ant*Nfreqs+readout_freq

    grad0 = dwcal.jac_dw_cal(
        gains0_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    grad1 = dwcal.jac_dw_cal(
        gains1_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    hess = dwcal.hess_dw_cal(
        gains_init_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    empirical_value = (
        np.real(grad1[readout_ind])
        - np.real(grad0[readout_ind])
    ) / delta_gains
    calc_value = hess[test_ind, readout_ind]
    print(f"Empirical value: {empirical_value}")
    print(f"Calculated value: {calc_value}")


def test_cost_func_calculations():

    data, model = dwcal.get_test_data(
        model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
        model_use_model=True,
        data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        debug_limit_freqs=None,
        use_antenna_list=[3, 4, 57, 70, 92, 110],
        use_flagged_baselines=False,
    )

    Nants = data.Nants_data
    Nbls = data.Nbls
    Ntimes = data.Ntimes
    Nfreqs = data.Nfreqs

    # Format visibilities
    data_visibilities = np.zeros((Ntimes, Nbls, Nfreqs), dtype=complex)
    model_visibilities = np.zeros((Ntimes, Nbls, Nfreqs), dtype=complex)
    flag_array = np.zeros((Ntimes, Nbls, Nfreqs), dtype=bool)
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
        flag_array[time_ind, :, :] = np.max(
            np.stack(
                [
                    np.squeeze(model_copy.flag_array, axis=(1, 3)),
                    np.squeeze(data_copy.flag_array, axis=(1, 3)),
                ]
            ),
            axis=0,
        )

    if not np.max(flag_array):  # Check for flags
        apply_flags = False

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

    # Initialize gains
    gain_init_noise = 0.1
    gains_init = np.random.normal(
        1.0, gain_init_noise, size=(Nants, Nfreqs),
    ) + 1.0j * np.random.normal(0.0, gain_init_noise, size=(Nants, Nfreqs),)
    # Expand the initialized values
    x0 = np.stack((np.real(gains_init), np.imag(gains_init)), axis=0).flatten()

    cov_mat = dwcal.get_weighted_cov_mat(
        Nfreqs, Nbls, metadata_reference.uvw_array, metadata_reference.freq_array
    )

    test_ant = 3
    test_freq = 1
    readout_ant = 2
    readout_freq = 2
    delta_gains = 0.0001

    test_grad_real(
        test_ant,
        test_freq,
        delta_gains,
        gains_init,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    test_grad_imag(
        test_ant,
        test_freq,
        delta_gains,
        gains_init,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )

    test_hess_real_real(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
        delta_gains,
        gains_init,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        cov_mat,
        data_visibilities,
    )


if __name__ == "__main__":
    test_cost_func_calculations()
