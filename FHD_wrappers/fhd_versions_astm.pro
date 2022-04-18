pro fhd_versions_astm
  except=!except
  !except=0
  heap_gc

  args = Command_Line_Args(count=nargs)
  output_directory = args[0]
  version = args[1]
  vis_file_list = args[2]

  case version of

    'rlb_test_run_Aug2021': begin
      catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
    end

    'rlb_model_GLEAM_Aug2021': begin
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

    'rlb_model_GLEAM_bright_sources_Aug2021': begin
      recalculate_all = 0
      in_situ_sim_input = '/lustre/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Aug2021/vis_data'
      return_cal_visibilities = 0
      catalog_file_path = 0
      diffuse_calibrate = 0
      diffuse_model = 0
      model_visibilities = 1
      model_catalog_file_path = '/opt/astro/devel/rbyrne/rlb_LWA/GLEAM_bright_sources.sav'
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

    'rlb_test_LWA_beam': begin
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      beam_model_version = 0
      unflag_all = 1
      calibrate_visibilities = 0
      model_visibilities = 0
      snapshot_healpix_export = 0
    end

    'rlb_test_LWA_increase_uv_resolution': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015_new.fits'
      beam_model_version = 0
      calibrate_visibilities = 0
      model_visibilities = 0
      snapshot_healpix_export = 0
      dimension = 2000
      kbinsize = 0.2
    end

    'rlb_test_LWA_short_baselines': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015_new.fits'
      beam_model_version = 0
      calibrate_visibilities = 0
      model_visibilities = 0
      snapshot_healpix_export = 0
      max_baseline = 50
      dimension = 2500
      kbinsize = 0.04
    end

    'rlb_test_LWA_rebase_Feb2022': begin
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      beam_model_version = 0
      calibrate_visibilities = 0
      model_visibilities = 0
      snapshot_healpix_export = 0
    end

    'rlb_LWA_imaging_Apr2022': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/opt/astro/devel/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      ; Try to force per-frequency calibration
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      ; Added Apr 12:
      ; flag_calibration = 0 ;Try turning off calibration flagging
      snapshot_healpix_export = 0 ;Healpix export does not work with just one time step
      ; Added Apr 15:
      min_cal_baseline = 0
    end

    'rlb_LWA_imaging_average_calibration_Apr2022': begin
      recalculate_all = 0
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/opt/astro/devel/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      ; Try to force per-frequency calibration
      bandpass_calibrate = 1
      cal_amp_degree_fit = 0 ;Average gain amplitudes across frequency
      cable_bandpass_fit = 0 ;Do not fit cable reflections
      cal_mode_fit = 0 ;Do not fit cable reflections
      cal_reflection_mode_theory = 0 ;Do not fit cable reflections
      calibration_polyfit = 1 ;Do frequency averaging
      cal_phase_degree_fit = 0 ;Average gain phases across frequency
    end

    'rlb_LWA_image_calibrated_data_Apr2022': begin
      recalculate_all = 0
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 2
    end

    'rlb_LWA_imaging_optimal_Apr2022': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/opt/astro/devel/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0 ;Healpix export does not work with just one time step
      min_cal_baseline = 0
      image_filter_fn = "filter_uv_optimal"
    end

    'rlb_LWA_model_mmode_map_Apr2022': begin
      recalculate_all = 0
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/opt/astro/devel/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 2
      model_visibilities = 1
      model_catalog_file_path = '/opt/astro/devel/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      model_subtract_sidelobe_catalog  = '/opt/astro/devel/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_model_sources = 1
      diffuse_model = "/lustre/rbyrne/ovro_lwa_sky_map_73.152MHz.skyh5"
      diffuse_units_kelvin = 1
      return_cal_visibilities = 0
      image_filter_fn = "filter_uv_optimal"
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
