pro normalize_uvf_from_kpar0

  reference_run_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_process_uv_density_sims_May2023"
  unnormalized_run_path = "/safepool/rbyrne/fhd_outputs/fhd_rlb_uv_density_sims_beam_error_normalized_May2023"
  obsid = "sim_uv_spacing_10_short_bls"
  pol = "xx"
  npols = 2
  nfreqs = 192
  
  reference_kpar0_path = reference_run_path+'/ps/data/1d_binning/'+obsid+'__gridded_uvf_noimgclip_dirty_'+pol+'_dft_averemove_swbh_dencorr_k0power.idlsave'
  unnormalized_kpar0_path = unnormalized_run_path+'/ps/data/1d_binning/'+obsid+'__gridded_uvf_noimgclip_dirty_'+pol+'_dft_averemove_swbh_dencorr_k0power.idlsave'
  
  reference_power = getvar_savefile(reference_kpar0_path, 'power')
  unnormalized_power = getvar_savefile(unnormalized_kpar0_path, 'power')
  
  norm_factor = mean(reference_power/unnormalized_power, /nan)
  print, "Normalization factor: " + string(norm_factor)
  
  uvf_cube_path = unnormalized_run_path+"/"+obsid+"__gridded_uvf.sav"
  dirty_uv_arr = getvar_savefile(uvf_cube_path, "DIRTY_UV_ARR")
  weights_uv_arr = getvar_savefile(uvf_cube_path, "WEIGHTS_UV_ARR")
  variance_uv_arr = getvar_savefile(uvf_cube_path, "VARIANCE_UV_ARR")
  obs_out = getvar_savefile(uvf_cube_path, "OBS_OUT")
  
  total_power_prev = 0
  for pol_ind=0,npols-1 do begin
    for freq_ind=0,nfreqs-1 do begin
      total_power_prev += total(abs(*dirty_uv_arr[pol_ind, freq_ind]))
    endfor
  endfor
  print, "Total uvf power, initial: " + string(total_power_prev)

  for pol_ind=0,npols-1 do begin
    for freq_ind=0,nfreqs-1 do begin
      *dirty_uv_arr[pol_ind, freq_ind] *= sqrt(norm_factor)
    endfor
  endfor
  
  total_power_new = 0
  for pol_ind=0,npols-1 do begin
    for freq_ind=0,nfreqs-1 do begin
      total_power_new += total(abs(*dirty_uv_arr[pol_ind, freq_ind]))
    endfor
  endfor
  print, "Total uvf power, normalized: " + string(total_power_new)

  save, dirty_uv_arr, obs_out, variance_uv_arr, weights_uv_arr, filename=uvf_cube_path

end