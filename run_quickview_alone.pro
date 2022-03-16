pro run_quickview_alone

  fhd_path = '/Users/ruby/Astro/polarized_source_sims_Feb2022/fhd_rlb_polarized_source_sim_optimal_weighting_Mar2021'
  obsid = 'polarized_source_MWA_sim_results'
  obs = getvar_savefile(fhd_path+'/metadata/'+obsid+'_obs.sav', 'obs')
  status_str = getvar_savefile(fhd_path+'/metadata/'+obsid+'_status.sav', 'status_str')
  psf = getvar_savefile(fhd_path+'/beams/'+obsid+'_beams.sav', 'psf')
  cal = 0
  jones = getvar_savefile(fhd_path+'/beams/'+obsid+'_jones.sav', 'jones')
  skymodel = getvar_savefile(fhd_path+'/output_data/'+obsid+'_skymodel.sav', 'skymodel')
  fhd_params = getvar_savefile(fhd_path+'/metadata/'+obsid+'_params.sav', 'params')
  image_uv_arr = ptrarr(4)
  im_uv_xx = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_XX.sav', 'grid_uv')
  im_uv_yy = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_YY.sav', 'grid_uv')
  im_uv_xy = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_XY.sav', 'grid_uv')
  im_uv_yx = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_YX.sav', 'grid_uv')
  image_uv_arr[0] = ptr_new(im_uv_xx)
  image_uv_arr[1] = ptr_new(im_uv_yy)
  image_uv_arr[2] = ptr_new(im_uv_xy)
  image_uv_arr[3] = ptr_new(im_uv_yx)
  weights_arr = ptrarr(4)
  weights_xx = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_weights_XX.sav', 'weights_uv')
  weights_yy = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_weights_YY.sav', 'weights_uv')
  weights_xy = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_weights_XY.sav', 'weights_uv')
  weights_yx = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_weights_YX.sav', 'weights_uv')
  weights_arr[0] = ptr_new(weights_xx)
  weights_arr[1] = ptr_new(weights_yy)
  weights_arr[2] = ptr_new(weights_xy)
  weights_arr[3] = ptr_new(weights_yx)
  model_uv_arr = ptrarr(4)
  mod_uv_xx = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_model_XX.sav', 'grid_uv_model')
  mod_uv_yy = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_model_YY.sav', 'grid_uv_model')
  mod_uv_xy = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_model_XY.sav', 'grid_uv_model')
  mod_uv_yx = getvar_savefile(fhd_path+'/grid_data/'+obsid+'_uv_model_YX.sav', 'grid_uv_model')
  model_uv_arr[0] = ptr_new(mod_uv_xx)
  model_uv_arr[1] = ptr_new(mod_uv_yy)
  model_uv_arr[2] = ptr_new(mod_uv_xy)
  model_uv_arr[3] = ptr_new(mod_uv_yx)
  image_filter_fn = 'filter_uv_optimal'
  file_path_fhd = fhd_path+'/'+obsid

  ; Stokes limits need to be divided by 2 to account for FHD bug
  stokes_high = 7e5/2
  stokes_low = -3e5/2
  instr_high = stokes_high/8000
  instr_low = stokes_low/8000
  fhd_quickview,obs,status_str,psf,cal,jones,skymodel,fhd_params,image_uv_arr=image_uv_arr,weights_arr=weights_arr,$
    model_uv_arr=model_uv_arr,image_filter_fn=image_filter_fn,file_path_fhd=file_path_fhd,$
    mark_zenith = 1, stokes_high = stokes_high, stokes_low = stokes_low, instr_high = instr_high, instr_low = instr_low

end