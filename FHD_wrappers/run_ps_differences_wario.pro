pro run_ps_differences_wario

  ;ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Nov2022"], $
  ;  ["random_gains_diagonal", "random_gains_dwcal"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar, $
  ;  data_min_abs=[1e1,1e12]
  ;ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Nov2022", "/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ;  ["random_gains_diagonal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar, $
  ;  data_min_abs=[1e1,1e12]
  ;ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Nov2022", "/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ;  ["random_gains_dwcal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar, $
  ;  data_min_abs=[1e1,1e12]

  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], $
    "sim_uv_spacing_10_short_bls", /uvf_input, /png, /refresh_diff, pols=["xx"]

end
