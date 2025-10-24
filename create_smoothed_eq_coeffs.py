import numpy as np
import scipy.io

eq_coeffs_mat = scipy.io.loadmat("/home/pipeline/opsdata/20241213-settingsAll-day.mat")  # or *night.mat

eq_coeffs_new = np.zeros((7, 512*8), dtype=float)
for ind in range(7):
    eq_coeffs = eq_coeffs_mat["coef"][ind, :]
    eq_coeffs = np.repeat(eq_coeffs, 8)
    gaussian = scipy.signal.windows.gaussian(len(eq_coeffs), 8)
    eq_coeffs_smoothed = np.copy(eq_coeffs)
    for eq_ind in range(len(eq_coeffs)):
        start_ind = eq_ind - int(np.floor(len(gaussian)/2))
        end_ind = start_ind + len(gaussian)
        gaussian_start_ind = 0
        gaussian_end_ind = len(gaussian)
        if start_ind < 0:
            gaussian_start_ind = -1 * start_ind
            start_ind = 0
        if end_ind > len(gaussian):
            gaussian_end_ind = len(gaussian) - (end_ind - len(gaussian))
            end_ind = len(gaussian)
        smoothing_func = np.copy(gaussian[gaussian_start_ind:gaussian_end_ind])
        smoothing_func /= np.sum(smoothing_func)  # Normalize
        eq_coeffs_smoothed[eq_ind] = np.sum(eq_coeffs[start_ind:end_ind] * smoothing_func)
    eq_coeffs_new[ind, :] = eq_coeffs_smoothed

eq_coeffs_mat["coef"] = eq_coeffs_new
scipy.io.savemat("/lustre/rbyrne/20250612-settingsAll-day_smoothed.mat", eq_coeffs_mat)
