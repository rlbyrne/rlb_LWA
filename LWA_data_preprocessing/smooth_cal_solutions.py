import pyuvdata
import numpy as np
import scipy


def gain_phase_res_func(x, gains, freq_array, branch_cut_loc):
    fit_value = np.zeros_like(freq_array, dtype=float)
    for x_ind, x_val in enumerate(x):
        fit_value += x_val * freq_array**x_ind
    fit_value = branch_cut(fit_value, branch_cut_loc=branch_cut_loc)
    gain_phase = branch_cut(np.angle(gains), branch_cut_loc=branch_cut_loc)
    res = np.sum((fit_value - gain_phase) ** 2)
    return res


def branch_cut(phases, branch_cut_loc=np.pi):

    phases -= branch_cut_loc
    phases %= 2 * np.pi
    phases += branch_cut_loc
    return phases


def find_optimal_branch_cut_loc(phases):

    phases = branch_cut(phases, branch_cut_loc=0)
    bin_edges = np.linspace(-np.pi, np.pi, num=100)
    hist, bin_edges = np.histogram(phases, bins=bin_edges)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    empty_bin_locs = bin_centers[np.where(hist == np.min(hist))[0]]
    if len(empty_bin_locs) == 1:
        return empty_bin_locs[0]
    else:
        return np.atan2(
            np.mean(np.sin(empty_bin_locs)), np.mean(np.cos(empty_bin_locs))
        )


def gain_phase_fit_search(gain_phases, freq_array, branch_cut_loc=np.pi):

    gain_phase = branch_cut(gain_phases, branch_cut_loc=branch_cut_loc)
    mean_freq = np.mean(freq_array)
    freqs_mean_subtracted = freq_array - mean_freq

    y_intercept = gain_phase[
        np.where(
            np.abs(freqs_mean_subtracted) == np.min(np.abs(freqs_mean_subtracted))
        )[0][0]
    ]
    channel_width = (np.max(freq_array) - np.min(freq_array)) / len(freq_array)
    check_slope_maximum = (
        np.pi / 10
    ) / channel_width  # Shouldn't be too big to prevent excessive wrapping
    check_slopes = np.linspace(-check_slope_maximum, check_slope_maximum, num=501)
    res_array = np.zeros_like(check_slopes, dtype=float)
    for slope_ind, slope in enumerate(check_slopes):
        res_array[slope_ind] = gain_phase_res_func(
            np.array([y_intercept, slope]), gains, freqs_mean_subtracted, branch_cut_loc
        )
    best_fit_slope = check_slopes[np.where(res_array == np.min(res_array))[0][0]]
    y_intercept -= best_fit_slope * mean_freq
    return np.array([y_intercept, best_fit_slope])


def calculate_smoothed_solutions(cal, freq_array_hz, amp_deg=2, phase_deg=1):

    cal.select(frequencies=freq_array_hz)

    amp_fit = np.zeros((amp_deg + 1, cal.Nants_data, cal.Ntimes, cal.Njones))
    phase_fit = np.zeros((phase_deg + 1, cal.Nants_data, cal.Ntimes, cal.Njones))
    for ant_ind in range(cal.Nants_data):
        for time_ind in range(cal.Ntimes):
            for pol_ind in range(cal.Njones):

                flags = cal.flag_array[ant_ind, :, time_ind, pol_ind]
                if np.min(flags):  # All flagged
                    continue
                gains = cal.gain_array[ant_ind, :, time_ind, pol_ind]
                gains = gains[np.where(~flags)]
                freqs_use = freq_array_hz[np.where(~flags)]

                amp_fit[:, ant_ind, time_ind, pol_ind] = np.polyfit(
                    freqs_use, np.abs(gains), amp_deg
                )[::-1]

                branch_cut_loc = find_optimal_branch_cut_loc(np.angle(gains))
                phase_fit_starting_guess = gain_phase_fit_search(
                    np.angle(gains), freqs_use, branch_cut_loc=branch_cut_loc
                )
                gain_phases_residual = (
                    np.angle(gains)
                    - phase_fit_starting_guess[0]
                    - phase_fit_starting_guess[1] * freqs_use
                )
                if phase_deg == 0:
                    phase_fit_starting_guess = phase_fit_starting_guess[[0]]
                elif phase_deg > 1:
                    phase_fit_starting_guess.append(
                        np.zeros(phase_deg - 1, dtype=float)
                    )
                optimize_result = scipy.optimize.minimize(
                    gain_phase_res_func,
                    np.zeros(phase_deg + 1, dtype=float),
                    args=(gains, freqs_use, branch_cut_loc),
                    method="Powell",
                    tol=1e-6,
                )
                phase_fit[:, ant_ind, time_ind, pol_ind] = optimize_result.x

    return amp_fit, phase_fit


def apply_smoothing(cal, amp_fit, phase_fit, inplace=False):

    gains_smoothed = np.copy(cal.gain_array)
    for ant_ind in range(cal.Nants_data):
        for time_ind in range(cal.Ntimes):
            for pol_ind in range(cal.Njones):
                gain_amps = np.zeros_like(cal.freq_array)
                for deg, amp_fit_val in enumerate(
                    amp_fit[:, ant_ind, time_ind, pol_ind]
                ):
                    gain_amps += amp_fit_val * cal.freq_array**deg
                gain_phases = np.zeros_like(cal.freq_array)
                for deg, phase_fit_val in enumerate(
                    phase_fit[:, ant_ind, time_ind, pol_ind]
                ):
                    gain_phases += phase_fit_val * cal.freq_array**deg
                gains_smoothed[ant_ind, :, time_ind, pol_ind] = gain_amps * np.exp(
                    1j * gain_phases
                )

    if inplace:
        cal.gain_array = gains_smoothed
    else:
        cal_smoothed = cal.copy()
        cal_smoothed.gain_array = gains_smoothed
        return cal_smoothed


def smooth_cal(cal, amp_deg=2, phase_deg=1, freq_array_hz=None):

    if freq_array_hz is None:
        freq_array_hz = cal.freq_array

    amp_fit, phase_fit = calculate_smoothed_solutions(
        cal, freq_array_hz, amp_deg=amp_deg, phase_deg=phase_deg
    )

    cal_smoothed = apply_smoothing(cal, amp_fit, phase_fit)
    return cal_smoothed
