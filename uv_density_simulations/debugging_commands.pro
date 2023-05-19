dirty_uv_arr_new = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Apr2023_6/sim_uv_spacing_1__gridded_uvf.sav", "dirty_uv_arr")
dirty_uv_arr_old = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Mar2023/sim_uv_spacing_1__gridded_uvf.sav", "dirty_uv_arr")

variance_uv_arr_new = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Apr2023_6/sim_uv_spacing_1__gridded_uvf.sav", "variance_uv_arr")
variance_uv_arr_old = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Mar2023/sim_uv_spacing_1__gridded_uvf.sav", "variance_uv_arr")

weights_uv_arr_new = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Apr2023_6/sim_uv_spacing_1__gridded_uvf.sav", "weights_uv_arr")
weights_uv_arr_old = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Mar2023/sim_uv_spacing_1__gridded_uvf.sav", "weights_uv_arr")

dirty_diff_array = make_array(2, 192, 1200, 1200, /dcomplex, value=0.)
for pol_ind=0,1 do for freq_ind=0,191 do dirty_diff_array[pol_ind, freq_ind, *, *] = *dirty_uv_arr_new[pol_ind, freq_ind] - *dirty_uv_arr_old[pol_ind, freq_ind]

variance_diff_array = make_array(2, 192, 1200, 1200, /dcomplex, value=0.)
for pol_ind=0,1 do for freq_ind=0,191 do variance_diff_array[pol_ind, freq_ind, *, *] = *variance_uv_arr_new[pol_ind, freq_ind] - *variance_uv_arr_old[pol_ind, freq_ind]

weights_diff_array = make_array(2, 192, 1200, 1200, /dcomplex, value=0.)
for pol_ind=0,1 do for freq_ind=0,191 do weights_diff_array[pol_ind, freq_ind, *, *] = *weights_uv_arr_new[pol_ind, freq_ind] - *weights_uv_arr_old[pol_ind, freq_ind]

obs_new = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Apr2023_6/sim_uv_spacing_1__gridded_uvf.sav", "obs_out")
obs_old = getvar_savefile("/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_Mar2023/sim_uv_spacing_1__gridded_uvf.sav", "obs_out")

; Debugging the power normalization
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_normalized_May2023", "sim_uv_spacing_10_short_bls", /png, /plot_kpar_power, /refresh_ps, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_info
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_normalized_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_10_short_bls"], /uvf_input, /png, /refresh_diff

; Difference calibration error:
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_10_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_10_short_bls_cal_error", "sim_uv_spacing_10_short_bls"], /uvf_input, /refresh_diff, /png
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_5_short_bls_cal_error", "sim_uv_spacing_5_short_bls"], /uvf_input, /refresh_diff, /png
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_1_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_1_short_bls_cal_error", "sim_uv_spacing_1_short_bls"], /uvf_input, /refresh_diff, /png
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_0.5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_0.5_short_bls_cal_error", "sim_uv_spacing_0.5_short_bls"], /uvf_input, /refresh_diff, /png

; Modified gridding kernel commands
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_10_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_1_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_0.5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]

; Modified gridding kernel, gain amplitude error only, commands
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_10_short_bls_cal_amp_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_5_short_bls_cal_amp_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_1_short_bls_cal_amp_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_modified_kernel_May2023", "sim_uv_spacing_0.5_short_bls_cal_amp_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
