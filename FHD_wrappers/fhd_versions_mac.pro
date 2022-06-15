pro fhd_versions_mac
  except=!except
  !except=0
  heap_gc

  version = 'rlb_cal_sims_Apr2022'
  output_directory = '/Users/ruby/Astro/FHD_outputs'
  
  case version of
    
    'rlb_test_run_Jul2021': begin
      obs_id = '1061316296'
      vis_file_list = '/Users/ruby/Astro/1061316296.uvfits'
      n_pol=4
    end

    'rlb_model_GLEAM_Aug2021': begin
      obs_id = '1061316296'
      vis_file_list = '/Users/ruby/Astro/1061316296.uvfits'
      recalculate_all = 0
      return_cal_visibilities = 0
      catalog_file_path = 0
      diffuse_calibrate = 0
      diffuse_model = 0
      model_visibilities = 1
      model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      calibrate_visibilities = 0
      rephase_weights = 0
      restrict_hpx_inds = 0
      hpx_radius = 15
      return_sidelobe_catalog = 1
      dft_threshold = 0
      ring_radius = 0
      write_healpix_fits = 1
      debug_region_grow = 0
      n_pol = 2
      unflag_all = 1  ; unflag for simulation
      beam_nfreq_avg = 384 ; use one beam for all frequencies
      save_uvf = 1
    end
    
    'rlb_test_HERA_beam': begin
      obs_id = '1061316296'
      vis_file_list = '/Users/ruby/Astro/1061316296.uvfits'
      instrument = 'hera'
      beam_model_version = 4
      unflag_all = 1
      calibrate_visibilities = 0
    end
    
    'rlb_test_LWA_beam': begin
      obs_id = '1061316296_small'
      vis_file_list = '/Users/ruby/Astro/'+obs_id+'.uvfits'
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/Users/ruby/Astro/LWA_beams/LWAbeam_2015.fits'
      beam_model_version = 0
      unflag_all = 1
      calibrate_visibilities = 0
    end
    
    'rlb_test_LWA_data_one_beam': begin
      recalculate_all = 1
      obs_id = '2019-11-21T23:00:08'
      vis_file_list = '/Users/ruby/Astro/LWA_data/'+obs_id+'.uvfits'
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/Users/ruby/Astro/LWA_beams/LWAbeam_2015_new.fits'
      beam_model_version = 0
      unflag_all = 1
      calibrate_visibilities = 0
      beam_nfreq_avg = 384
      psf_resolution = 8
      snapshot_healpix_export = 0
    end
    
    'rlb_test_LWA_data': begin
      recalculate_all = 1
      obs_id = '2019-11-21T23:00:08'
      vis_file_list = '/Users/ruby/Astro/LWA_data/'+obs_id+'.uvfits'
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/Users/ruby/Astro/LWA_beams/LWAbeam_2015_new.fits'
      beam_model_version = 0
      unflag_all = 1
      calibrate_visibilities = 0
      ;beam_nfreq_avg = 384
      psf_resolution = 8
      snapshot_healpix_export = 0
    end
    
    'rlb_test_LWA_data_rephased': begin
      recalculate_all = 1
      obs_id = '2019-11-21T23_00_08_rephased'
      vis_file_list = '/Users/ruby/Astro/LWA_data/'+obs_id+'.uvfits'
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/Users/ruby/Astro/LWA_beams/LWAbeam_2015_new.fits'
      beam_model_version = 0
      unflag_all = 0
      calibrate_visibilities = 0
      ;beam_nfreq_avg = 384
      ;psf_resolution = 8
      snapshot_healpix_export = 0
    end
    
    'rlb_test_skyh5_sky_model': begin
      recalculate_all = 1
      export_images = 1
      model_visibilities = 1
      model_catalog_file_path = '/Users/ruby/Astro/polarized_source_sims_Feb2022/polarized_source.skyh5'
      ;obs_id = '1061316296_small'
      obs_id = 'polarized_source_MWA_sim_results'
      ;vis_file_list = '/Users/ruby/Astro/'+obs_id+'.uvfits'
      vis_file_list = '/Users/ruby/Astro/polarized_source_sims_Feb2022/'+obs_id+'.uvfits'
      calibrate_visibilities = 0
      model_visibilities = 1
      ;model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      snapshot_healpix_export = 0
      n_pol = 4
      image_filter_fn = "filter_uv_optimal"
      instr_high = 15000
      instr_low = -15000
      stokes_high = 2e6
      stokes_low = -2e6
      mark_zenith = 1
    end
    
    'rlb_LWA_imaging_Apr2022': begin
      obs_id = '20220307_175923_61MHz_uncalib'
      vis_file_list = '/Users/ruby/Astro/LWA_data/LWA_data_20220307/'+obs_id+'.uvfits'
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/Users/ruby/Astro/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/Users/ruby/Astro/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      ; Try to force per-frequency calibration
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      flag_calibration = 0 ;Try turning off calibration flagging
      snapshot_healpix_export = 0 ;Healpix export does not work with just one time step
    end
    
    'rlb_cal_sims_Apr2022': begin
      obs_id = 'vanilla_cal'
      vis_file_list = '/Users/ruby/Astro/dwcal_tests_Apr2022/caltest_Apr12/'+obs_id+'.uvfits'
      recalculate_all = 1
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      model_subtract_sidelobe_catalog = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      allow_sidelobe_model_sources = 1
      unflag_all = 1
      beam_nfreq_avg = 384
      n_pol = 1
      export_images = 0 ;Cannot export images with just one polarization
      save_uvf = 1
    end
    
    'rlb_test_max_baseline': begin
      recalculate_all = 1
      obs_id = '1061316296_small'
      vis_file_list = '/Users/ruby/Astro/1061316296_small.uvfits'
    end
    
  endcase

  undefine, uvfits_subversion, uvfits_version

  fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory)
  healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version)

  ; Set global defaults and bundle all the variables into a structure.
  ; Any keywords set on the command line or in the top-level wrapper will supercede these defaults
  eor_wrapper_defaults,extra
  fhd_depreciation_test, _Extra=extra

  print,""
  print,"Keywords set in wrapper:"
  print,structure_to_text(extra)
  print,""

  general_obs,_Extra=extra

end
