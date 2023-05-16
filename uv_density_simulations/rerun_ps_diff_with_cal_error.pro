pro rerun_ps_diff_with_cal_error

  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023", "sim_uv_spacing_10_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_10_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_10_short_bls_cal_error", "sim_uv_spacing_10_short_bls"], /uvf_input, /refresh_diff, /png, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]

  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023", "sim_uv_spacing_5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_5_short_bls_cal_error", "sim_uv_spacing_5_short_bls"], /uvf_input, /refresh_diff, /png, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]

  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023", "sim_uv_spacing_1_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_1_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_1_short_bls_cal_error", "sim_uv_spacing_1_short_bls"], /uvf_input, /refresh_diff, /png, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]

  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023", "sim_uv_spacing_0.5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "sim_uv_spacing_0.5_short_bls_cal_error", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"], /no_evenodd, /refresh_ps, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]
  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_cal_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], ["sim_uv_spacing_0.5_short_bls_cal_error", "sim_uv_spacing_0.5_short_bls"], /uvf_input, /refresh_diff, /png, kperp_range_lambda_1dave=[1, 45], kperp_range_lambda_kparpower=[1, 45]

end
