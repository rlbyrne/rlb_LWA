import delay_weighted_cal as dwcal
import numpy as np


def test_grad(
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
    real_part=True,
):

    print("*******")
    if real_part:
        print("Testing the gradient calculation, real part")
        multiplier = 1.0
    else:
        print("Testing the gradient calculation, imaginary part")
        multiplier = 1j

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= multiplier * delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += multiplier * delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant * Nfreqs + test_freq
    if not real_part:
        test_ind += Nants * Nfreqs

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


def test_hess(
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
    real_part1=True,
    real_part2=True,
):

    if real_part1:
        part1_text = "real"
        multiplier = 1.0
    else:
        part1_text = "imag"
        multiplier = 1j
    if real_part2:
        part2_text = "real"
    else:
        part2_text = "imag"

    print("*******")
    print(f"Testing the hessian calculation, {part1_text}-{part2_text} part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= multiplier * delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += multiplier * delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant * Nfreqs + test_freq
    if not real_part1:
        test_ind += Nants * Nfreqs
    readout_ind = readout_ant * Nfreqs + readout_freq
    if not real_part2:
        readout_ind += Nants * Nfreqs

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

    empirical_value = (grad1[readout_ind] - grad0[readout_ind]) / delta_gains
    calc_value = hess[test_ind, readout_ind]
    print(f"Empirical value: {empirical_value}")
    print(f"Calculated value: {calc_value}")


def test_hess_real_imag(
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
    print("Testing the hessian calculation, real-imag part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant * Nfreqs + test_freq
    readout_ind = Nfreqs * Nants + readout_ant * Nfreqs + readout_freq

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

    empirical_value = (grad1[readout_ind] - grad0[readout_ind]) / delta_gains
    calc_value = hess[test_ind, readout_ind]
    print(f"Empirical value: {empirical_value}")
    print(f"Calculated value: {calc_value}")


def test_derivative_calculations():

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
    # gain_init_noise = 0.1
    # gains_init = np.random.normal(
    #    1.0, gain_init_noise, size=(Nants, Nfreqs),
    # ) + 1.0j * np.random.normal(0.0, gain_init_noise, size=(Nants, Nfreqs),)
    gains_init = np.full((Nants, Nfreqs), 1.01 + 0.01j, dtype="complex")

    cov_mat = dwcal.get_weighted_cov_mat(
        Nfreqs, Nbls, metadata_reference.uvw_array, metadata_reference.freq_array
    )

    test_ant = 3
    test_freq = 1
    readout_ant = 2
    readout_freq = 1
    delta_gains = 0.0001

    test_grad(
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
        real_part=True,
    )

    test_grad(
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
        real_part=False,
    )

    test_hess(
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
        real_part1=True,
        real_part2=True,
    )

    test_hess(
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
        real_part1=True,
        real_part2=False,
    )

    test_hess(
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
        real_part1=False,
        real_part2=True,
    )

    test_hess(
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
        real_part1=False,
        real_part2=False,
    )


def test_derivative_calculations_randomized():

    Nants = 10
    Nbls = int((Nants ** 2 - Nants) / 2)
    Ntimes = 2
    Nfreqs = 384

    ant_1_array = np.zeros(Nbls, dtype=int)
    ant_2_array = np.zeros(Nbls, dtype=int)
    ind = 0
    for ant_1 in range(Nants):
        for ant_2 in range(ant_1 + 1, Nants):
            ant_1_array[ind] = ant_1
            ant_2_array[ind] = ant_2
            ind += 1

    # Format visibilities
    data_stddev = 6.0
    data_visibilities = np.random.normal(
        0.0, data_stddev, size=(Ntimes, Nbls, Nfreqs),
    ) + 1.0j * np.random.normal(0.0, data_stddev, size=(Ntimes, Nbls, Nfreqs),)
    model_visibilities = np.random.normal(
        0.0, data_stddev, size=(Ntimes, Nbls, Nfreqs),
    ) + 1.0j * np.random.normal(0.0, data_stddev, size=(Ntimes, Nbls, Nfreqs),)

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((Nbls, Nants), dtype=int)
    gains_exp_mat_2 = np.zeros((Nbls, Nants), dtype=int)
    antenna_list = np.unique([ant_1_array, ant_2_array])
    for baseline in range(Nbls):
        gains_exp_mat_1[baseline, np.where(antenna_list == ant_1_array[baseline])] = 1
        gains_exp_mat_2[baseline, np.where(antenna_list == ant_2_array[baseline])] = 1

    # Initialize gains
    gain_init_noise = 0.1
    gains_init = np.random.normal(
        1.0, gain_init_noise, size=(Nants, Nfreqs),
    ) + 1.0j * np.random.normal(0.0, gain_init_noise, size=(Nants, Nfreqs),)

    cov_mat_stddev = 5.0
    cov_mat = np.random.normal(0.0, cov_mat_stddev, size=(Nbls, Nfreqs, Nfreqs))
    cov_mat += np.transpose(cov_mat, (0, 2, 1))  # Matrix must be Hermitian

    test_ant = np.random.randint(0, Nants - 1)
    test_freq = np.random.randint(0, Nfreqs - 1)
    readout_ant = np.random.randint(0, Nants - 1)
    while readout_ant == test_ant:  # Autocorrelations are excluded
        readout_ant = np.random.randint(0, Nants - 1)
    readout_freq = np.random.randint(0, Nfreqs - 1)
    delta_gains = 0.0001

    test_grad(
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
        real_part=True,
    )

    test_grad(
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
        real_part=False,
    )

    test_hess(
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
        real_part1=True,
        real_part2=True,
    )

    test_hess(
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
        real_part1=True,
        real_part2=False,
    )

    test_hess(
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
        real_part1=False,
        real_part2=True,
    )

    test_hess(
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
        real_part1=False,
        real_part2=False,
    )

    # Test hess frequency diagonals
    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=True,
        real_part2=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=True,
        real_part2=False,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=False,
        real_part2=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=False,
        real_part2=False,
    )


if __name__ == "__main__":
    test_derivative_calculations_randomized()
