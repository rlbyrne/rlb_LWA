pro fhd_versions_wario
  except=!except
  !except=0
  heap_gc

  args = Command_Line_Args(count=nargs)
  output_directory = args[0]
  version = args[1]
  vis_file_list = args[2]

  case version of

    'rlb_model_GLEAM_Dec2021': begin
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

    'rlb_model_GLEAM_bright_sources_Dec2021': begin
      recalculate_all = 0
      in_situ_sim_input = '/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Dec2021/vis_data'
      return_cal_visibilities = 0
      catalog_file_path = 0
      diffuse_calibrate = 0
      diffuse_model = 0
      model_visibilities = 1
      model_catalog_file_path = '/home/rbyrne/rlb_LWA/GLEAM_bright_sources.sav'
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

    'rlb_LWA_phasing_sim_Feb2021': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/safepool/rbyrne/LWA_pyuvsim_simulations/LWAbeam_2015_new.fits'
      beam_model_version = 0
      model_visibilities = 0
      calibrate_visibilities = 0
      snapshot_healpix_export = 0
      instr_high = 1e10
      instr_low = -1e10
      stokes_high = 1e10
      stokes_low = -1e10
    end

    'rlb_polarized_source_sim_Mar2021': begin
      recalculate_all = 0
      export_images = 1
      instrument = 'mwasim'
      import_pyuvdata_beam_filepath = '/safepool/rbyrne/pyuvsim_sims_for_polarimetry_paper/mwa_full_embedded_element_pattern.fits'
      model_visibilities = 0
      calibrate_visibilities = 0
      snapshot_healpix_export = 0
      n_pol = 4
    end

    'rlb_polarized_source_sim_optimal_weighting_Mar2021': begin
      recalculate_all = 0
      snapshot_recalculate = 1
      export_images = 1
      instrument = 'mwasim'
      import_pyuvdata_beam_filepath = '/safepool/rbyrne/pyuvsim_sims_for_polarimetry_paper/mwa_full_embedded_element_pattern.fits'
      model_visibilities = 1
      model_catalog_file_path = '/safepool/rbyrne/pyuvsim_sims_for_polarimetry_paper/polarized_source.skyh5'
      calibrate_visibilities = 0
      return_cal_visibilities = 0 ;Is needed because calibrate_visibilities is unset
      snapshot_healpix_export = 0
      n_pol = 4
      image_filter_fn = "filter_uv_optimal"
      instr_high = 1000
      instr_low = -1000
      stokes_high = 5e5
      stokes_low = -5e5
      mark_zenith = 1
    end

    'rlb_model_diffuse_skyh5_Mar2022': begin
      recalculate_all = 0
      calibrate_visibilities = 0
      return_cal_visibilities = 0  ; changed this for calibration transfer
      model_visibilities = 1
      model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      subtract_sidelobe_catalog = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      diffuse_model = '/safepool/rbyrne/diffuse_map.skyh5'
      n_pol = 4
      max_baseline = 50  ; use only baselines shorter than 50 wavelengths
      dimension = 208 ; limit the UV plane to regions that contain data
      image_filter_fn = 'filter_uv_optimal'
    end

    'rlb_model_GLEAM_Apr2022': begin
      recalculate_all = 0
      return_cal_visibilities = 0
      catalog_file_path = 0
      diffuse_calibrate = 0
      diffuse_model = 0
      model_visibilities = 1
      model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      model_subtract_sidelobe_catalog = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      allow_sidelobe_model_sources = 1
      calibrate_visibilities = 0
      rephase_weights = 0
      restrict_hpx_inds = 0
      hpx_radius = 15
      dft_threshold = 0
      ring_radius = 0
      n_pol = 2
      unflag_all = 1  ; unflag for simulation
      beam_nfreq_avg = 384 ; use one beam for all frequencies
    end

    'rlb_model_GLEAM_bright_sources_Apr2022': begin
      recalculate_all = 0
      in_situ_sim_input = '/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022/vis_data'
      return_cal_visibilities = 1
      catalog_file_path = '/home/rbyrne/rlb_LWA/GLEAM_bright_sources.sav'
      calibration_subtract_sidelobe_catalog = '/home/rbyrne/rlb_LWA/GLEAM_bright_sources.sav'
      allow_sidelobe_cal_sources = 1
      diffuse_calibrate = 0
      diffuse_model = 0
      calibrate_visibilities = 1
      rephase_weights = 0
      restrict_hpx_inds = 0
      hpx_radius = 15
      dft_threshold = 0
      ring_radius = 0
      n_pol = 2
      unflag_all = 1  ; unflag for simulation
      beam_nfreq_avg = 384 ; use one beam for all frequencies
      ; Force per-frequency calibration
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
    end

    'rlb_cal_sims_Apr2022': begin
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

    'rlb_cal_sims_Jun2022': begin
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

    'rlb_model_GLEAM_Jun2022': begin
      recalculate_all = 1
      return_cal_visibilities = 0
      catalog_file_path = 0
      diffuse_calibrate = 0
      diffuse_model = 0
      model_visibilities = 1
      model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      model_subtract_sidelobe_catalog = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      allow_sidelobe_model_sources = 1
      calibrate_visibilities = 0
      rephase_weights = 0
      restrict_hpx_inds = 0
      hpx_radius = 15
      dft_threshold = 0
      ring_radius = 0
      n_pol = 2
      unflag_all = 1  ; unflag for simulation
      flag_visibilities = 0  ; try turning this off
      beam_nfreq_avg = 384 ; use one beam for all frequencies
      max_baseline = 3000  ; try increasing max baseline to prevent baseline cutting
      dimension = 8192
    end

    'rlb_model_GLEAM_bright_sources_Jun2022': begin
      recalculate_all = 0
      in_situ_sim_input = '/safepool/rbyrne/fhd_outputs/fhd_rlb_model_GLEAM_Apr2022/vis_data'
      return_cal_visibilities = 1
      catalog_file_path = '/home/rbyrne/rlb_LWA/GLEAM_bright_sources.sav'
      calibration_subtract_sidelobe_catalog = '/home/rbyrne/rlb_LWA/GLEAM_bright_sources.sav'
      allow_sidelobe_cal_sources = 1
      diffuse_calibrate = 0
      diffuse_model = 0
      calibrate_visibilities = 1
      rephase_weights = 0
      restrict_hpx_inds = 0
      hpx_radius = 15
      dft_threshold = 0
      ring_radius = 0
      n_pol = 2
      unflag_all = 1  ; unflag for simulation
      flag_visibilities = 0
      beam_nfreq_avg = 384 ; use one beam for all frequencies
      ; Force per-frequency calibration
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      max_baseline = 3000  ; try increasing max baseline to prevent baseline cutting
      dimension = 8192
    end

    'rlb_cal_sims_2pol_Jun2022': begin
      recalculate_all = 1
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      model_catalog_file_path = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      model_subtract_sidelobe_catalog = filepath('GLEAM_v2_plus_rlb2019.sav',root=rootdir('FHD'),subdir='catalog_data')
      allow_sidelobe_model_sources = 1
      unflag_all = 1
      beam_nfreq_avg = 384
      n_pol = 2
      save_uvf = 1
    end

    'rlb_cal_sims_Nov2022': begin
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

    'rlb_cal_sims_old_FHD_Nov2022': begin
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

    'rlb_image_LWA_data_Nov2022': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 4
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      diffuse_units_kelvin = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      ;snapshot_healpix_export = 0
      min_cal_baseline = 0
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 1 ;allow calibration to flag antennas
      calibration_flag_iterate = 1 ;repeat calibration after flagging
      save_uvf = 1
    end

    'rlb_image_LWA_data_Jan2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 4
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      diffuse_units_kelvin = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_process_uv_density_sims_Feb2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Mar2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_image_LWA_data_Apr2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 4
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      diffuse_units_kelvin = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_VLSS_Apr2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 4
      calibration_catalog_file_path = '/home/rbyrne/FullVLSSCatalog.skyh5'
      allow_sidelobe_cal_sources = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_process_uv_density_sims_Mar2023_2': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Apr2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Apr2023_2': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Apr2023_3': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Apr2023_4': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Apr2023_5': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_Apr2023_6': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_May2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_uv_density_sims_beam_error_May2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_13m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_process_uv_density_sims_cal_error_May2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
      export_images = 0 ;for some reason image exporting is crashing
    end

    'rlb_process_uv_density_sims_cal_error_modified_kernel_May2023': begin
      recalculate_all = 1
      import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
      calibrate_visibilities = 0
      n_pol = 2
      snapshot_healpix_export = 1  ;required to activate save_uvf
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
      flag_visibilities = 0
      unflag_all = 1
      instrument = "ovro-lwa"
      beam_nfreq_avg = 1  ;do not average beam
      export_images = 0 ;for some reason image exporting is crashing
      ;kernel-related keywords
      kernel_window = 1
      debug_dim = 1
      beam_mask_threshold = 1e3
      interpolate_kernel = 1
    end

    'rlb_image_LWA_data_Jun2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 4
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_no_calibration_Jun2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_no_calibration_MWA_beam_Jun2023': begin
      recalculate_all = 1
      ;instrument = 'lwa'
      ;import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_no_calibration_all_baselines_Jun2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      min_baseline = 0
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_natural_weighting_Jun2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      min_baseline = 0
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_natural"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_natural_weighting_with_beam_speedup_Jun2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      min_baseline = 0
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_natural"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_test_jones_transpose_Jun2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      min_baseline = 0
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      ;min_cal_baseline = 30
      image_filter_fn = "filter_uv_natural"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_data_test_mwa_beam_import_Jun2023': begin
      recalculate_all = 1
      ;instrument = 'lwa'
      ;import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      min_baseline = 0
      ;calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      ;allow_sidelobe_cal_sources = 1
      ;diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      ;diffuse_units_kelvin = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      ;min_cal_baseline = 30
      image_filter_fn = "filter_uv_natural"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_no_calibration_Jul2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      model_visibilities = 0
      n_pol = 4
      min_baseline = 0
      snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_natural"
      save_uvf = 0
    end

    'rlb_image_LWA_data_Jul2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 4
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      snapshot_healpix_export = 0
      min_cal_baseline = 30
      image_filter_fn = "filter_uv_natural"
      flag_calibration = 0 ;allow calibration to flag antennas
      calibration_flag_iterate = 0 ;repeat calibration after flagging
      save_uvf = 0
    end

    'rlb_image_LWA_no_calibration_Aug2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      model_visibilities = 0
      n_pol = 4
      min_baseline = 0
      snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_natural"
      save_uvf = 0
    end

    'rlb_image_LWA_modified_kernel_Aug2023': begin
      recalculate_all = 0
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      model_visibilities = 0
      n_pol = 4
      min_baseline = 0
      snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      save_uvf = 0
      export_images = 1
      ;kernel-related keywords
      kernel_window = 1
      debug_dim = 1
      beam_mask_threshold = 1e3
      interpolate_kernel = 1
    end

    'rlb_image_LWA_modified_kernel_test_branch_Sept2023': begin
      recalculate_all = 0
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      model_visibilities = 0
      n_pol = 4
      min_baseline = 0
      snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      save_uvf = 0
      export_images = 1
      ;kernel-related keywords
      kernel_window = 1
      debug_dim = 1
      beam_mask_threshold = 1e3
      interpolate_kernel = 1
    end

    'rlb_LWA_image_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_LWA_model_diffuse_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      diffuse_model = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      diffuse_units_kelvin = 1
      model_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_LWA_caltest_cyg_cas_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      return_cal_visibilities = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      min_cal_baseline = 0
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 1 ;allow calibration to flag antennas
      calibration_flag_iterate = 1 ;repeat calibration after flagging
      save_uvf = 1
    end

    'rlb_LWA_caltest_mmode_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      diffuse_calibrate = "/safepool/rbyrne/transferred_from_astm/ovro_lwa_sky_map_73.152MHz.skyh5"
      diffuse_units_kelvin = 1
      allow_sidelobe_cal_sources = 1
      return_cal_visibilities = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      min_cal_baseline = 0
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 1 ;allow calibration to flag antennas
      calibration_flag_iterate = 1 ;repeat calibration after flagging
      save_uvf = 1
    end

    'rlb_LWA_caltest_cyg_cas_v2_branch_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      return_cal_visibilities = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      min_cal_baseline = 0
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 1 ;allow calibration to flag antennas
      calibration_flag_iterate = 1 ;repeat calibration after flagging
      save_uvf = 1
    end

    'rlb_LWA_test_diffuse_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      diffuse_model = "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial_Jy.skyh5"
      diffuse_units_kelvin = 0
      model_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_LWA_test_diffuse_Kelvin_conversion_Dec2023': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      diffuse_model = "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial.skyh5"
      diffuse_units_kelvin = 1
      model_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_LWA_test_diffuse_debug_Kelvin_conversion_Jan2024': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      diffuse_model = "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial.skyh5"
      diffuse_units_kelvin = 1
      model_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_LWA_test_diffuse_Jan2024': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      diffuse_model = "/safepool/rbyrne/skymodels/ovro_lwa_sky_map_73.152MHz_equatorial_Jy_per_sr.skyh5"
      diffuse_units_kelvin = 0
      model_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_LWA_caltest_cyg_cas_Jan2024': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 1
      n_pol = 2
      calibration_catalog_file_path = '/home/rbyrne/rlb_LWA/LWA_skymodels/cyg_cas.skyh5'
      allow_sidelobe_cal_sources = 1
      return_cal_visibilities = 1
      sim_over_calibrate = 1
      bandpass_calibrate = 0
      cable_bandpass_fit = 0
      cal_mode_fit = 0
      calibration_polyfit = 0
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      min_cal_baseline = 15  ;change this to match imaging tutorial
      image_filter_fn = "filter_uv_optimal"
      flag_calibration = 1 ;allow calibration to flag antennas
      calibration_flag_iterate = 1 ;repeat calibration after flagging
      save_uvf = 1
    end

    'rlb_LWA_generate_ps_Jan2024': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 0
      n_pol = 4
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
    end

    'rlb_vincent_sims_Mar2025_2': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/safepool/rbyrne/vincent_sims/airy_efield_10m.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 0
      n_pol = 2
      ;snapshot_healpix_export = 0
      image_filter_fn = "filter_uv_optimal"
      split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
      save_uvf = 1
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
