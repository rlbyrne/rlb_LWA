pro normalize_uvf_cubes
; This didn't actually work to correct the PS offset
; Use normalize_uvf_from_kpar0 instead

  reference_run_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"
  normalized_run_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_normalized_May2023"

  cube_name = "sim_uv_spacing_10_short_bls__gridded_uvf.sav"

  variance_uv_arr_ref = getvar_savefile(reference_run_path + "/" + cube_name, "VARIANCE_UV_ARR")
  variance_uv_arr = getvar_savefile(normalized_run_path + "/" + cube_name, "VARIANCE_UV_ARR")

  npols = 2
  nfreqs = 192
  ref_total_var = 0
  norm_total_var = 0
  for pol_ind=0,npols-1 do begin
    for freq_ind=0,nfreqs-1 do begin
      ref_total_var += total(abs(*variance_uv_arr_ref[pol_ind, freq_ind]))
      norm_total_var += total(abs(*variance_uv_arr[pol_ind, freq_ind]))
    endfor
  endfor

  norm_factor = sqrt(ref_total_var/norm_total_var)
  print, "Normalization factor: " + string(norm_factor)

  dirty_uv_arr = getvar_savefile(normalized_run_path + "/" + cube_name, "DIRTY_UV_ARR")
  weights_uv_arr = getvar_savefile(normalized_run_path + "/" + cube_name, "WEIGHTS_UV_ARR")
  obs_out = getvar_savefile(normalized_run_path + "/" + cube_name, "OBS_OUT")

  for pol_ind=0,npols-1 do begin
    for freq_ind=0,nfreqs-1 do begin
      *dirty_uv_arr[pol_ind, freq_ind] *= norm_factor
      *variance_uv_arr[pol_ind, freq_ind] *= norm_factor^2
      *weights_uv_arr[pol_ind, freq_ind] *= norm_factor
    endfor
  endfor

  save, dirty_uv_arr, obs_out, variance_uv_arr, weights_uv_arr, filename=normalized_run_path + "/" + cube_name

end
