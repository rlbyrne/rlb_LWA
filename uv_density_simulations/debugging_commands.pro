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

ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_normalized_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_10_short_bls"], /uvf_input, /png, /refresh_diff
