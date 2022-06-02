import numpy as np
import pyuvdata
import scipy.optimize
from dwcal import delay_weighted_cal as dwcal


def fft_visibilities(uv):
    delay_array = np.fft.fftfreq(uv.Nfreqs, d=uv.channel_width)
    delay_array = np.fft.fftshift(delay_array)
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft(uv.data_array, axis=2), axes=2))
    return fft_abs, delay_array


def cost_func_step_fit(x, diff_fft_abs_squared, delay_array, bl_lengths, c=3e8):
    wedge_error = x[0]
    window_error = x[1]
    boundary_slope = x[2] / c
    fit = np.full_like(diff_fft_abs_squared, window_error)
    for delay_ind, delay_val in enumerate(delay_array):
        wedge_bls = np.where(bl_lengths * boundary_slope > np.abs(delay_val))[0]
        if len(wedge_bls) > 0:
            fit[wedge_bls, delay_ind] = wedge_error
    return np.sum((diff_fft_abs_squared - fit) ** 2.0)


data, model = dwcal.get_test_data(
    model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_bright_sources_Apr2022",
    model_use_model=True,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Apr2022",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    use_autos=False,
    debug_limit_freqs=None,
    use_antenna_list=None,
    use_flagged_baselines=False,
)

diff_vis = data.diff_vis(model, inplace=False)

diff_fft_abs, delay_array = fft_visibilities(diff_vis)
bl_lengths = np.sqrt(np.sum(diff_vis.uvw_array**2.0, axis=1))

x0 = np.array([1.0, 1.0, 1.0])
step_fit_result = scipy.optimize.minimize(
    cost_func_step_fit,
    x0,
    args=(
        diff_fft_abs.squeeze() ** 2.0,
        delay_array,
        bl_lengths,
    ),
    method="Powell",
)

print(step_fit_result.message)
print(f"Wedge Variance: {step_fit_result.x[0]}")
print(f"Window Variance: {step_fit_result.x[1]}")
print(f"Wedge Boundary Slope: {step_fit_result.x[2]}")

# Results are:
# Wedge Variance: 7.060438044132621
# Window Variance: 0.09310922310592183
# Wedge Boundary Slope: 0.6284790822752272
