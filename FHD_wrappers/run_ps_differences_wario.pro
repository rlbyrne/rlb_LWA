ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["unity_gains_diagonal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["unity_gains_dwcal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["random_gains_diagonal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["random_gains_dwcal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["ripple_gains_diagonal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["ripple_gains_dwcal", "unity_gains_uncalib"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar

ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["unity_gains_diagonal", "unity_gains_dwcal"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["random_gains_diagonal", "random_gains_dwcal"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar
ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022"], $
  ["ripple_gains_diagonal", "ripple_gains_dwcal"], /uvf_input, /png, /refresh_diff, pols=["xx"], /invert_colorbar


;ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_cal_sims_Jun2022", "unity_gains_diagonal", /png, /plot_kpar_power, /uvf_input, pol_inc=["xx"]
