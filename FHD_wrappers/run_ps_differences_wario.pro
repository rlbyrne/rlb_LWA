pro run_ps_differences_wario

  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Nov2022"], $
    ["random_gains_diagonal", "random_gains_dwcal"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar, $
    data_min_abs=[1e1,1e12]
  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Nov2022", "/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
    ["random_gains_diagonal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar, $
    data_min_abs=[1e1,1e12]
  ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Nov2022", "/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
    ["random_gains_dwcal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar, $
    data_min_abs=[1e1,1e12]


  ;ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022", "unity_gains_diagonal", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"]

end
