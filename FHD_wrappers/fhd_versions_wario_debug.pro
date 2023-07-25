pro fhd_versions_wario_debug
  except=!except
  !except=0
  heap_gc

  output_directory = "/safepool/rbyrne/fhd_outputs"
  version = "rlb_image_LWA_no_calibration_debug_Jul2023"
  vis_file_list = "/safepool/rbyrne/lwa_data/newcal_testing_Jul2023/20230309_225134_73MHz_calibrated.uvfits"

  case version of

      'rlb_debug_uv_density_sims_May2023': begin
           recalculate_all = 1
           ;import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
           calibrate_visibilities = 0
           n_pol = 2
           snapshot_healpix_export = 1  ;required to activate save_uvf
           split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
           save_uvf = 1
           flag_visibilities = 0
           unflag_all = 1
           ;instrument = "ovro-lwa"
           beam_nfreq_avg = 1  ;do not average beam
       end

       'rlb_debug_uv_density_sims_May2023_2': begin
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

        'rlb_debug_modified_gridding_kernel_default_beam_May2023': begin
             recalculate_all = 1
             ;import_pyuvdata_beam_filepath = '/home/rbyrne/airy_14m.beamfits'
             calibrate_visibilities = 0
             n_pol = 2
             snapshot_healpix_export = 1  ;required to activate save_uvf
             split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
             save_uvf = 1
             flag_visibilities = 0
             unflag_all = 1
             ;instrument = "ovro-lwa"
             beam_nfreq_avg = 1  ;do not average beam
             ;kernel-related keywords
             kernel_window = 1
             debug_dim = 1
             beam_mask_threshold = 1e3
             interpolate_kernel = 1
         end

         'rlb_debug_modified_gridding_kernel_uvbeam_May2023': begin
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

          'rlb_image_LWA_no_calibration_debug_Jul2023': begin
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
