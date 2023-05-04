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

  uv_spacings = ["10", "5", "1", "0.5"]
  for uv_spacing_ind=0,n_elements(uv_spacings)-1 do begin
    uv_spacing = uv_spacings[uv_spacing_ind]
    ; Need to create info file by running ps_wrapper with refresh_info=0
    ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_May2023", "sim_uv_spacing_"+uv_spacing+"_short_bls", /png, /plot_kpar_power, refresh_ps=0, /uvf_input, pol_inc=["xx"], /no_evenodd
    ps_wrapper, "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023", "sim_uv_spacing_"+uv_spacing+"_short_bls", /png, /plot_kpar_power, refresh_ps=0, /uvf_input, pol_inc=["xx"], /no_evenodd
    ps_diff_wrapper, ["/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_May2023", "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"], $
      ["sim_uv_spacing_"+uv_spacing+"_short_bls"], /uvf_input, /png, /refresh_diff
  endfor

end
