pro fhd_versions_wario_debug
  except=!except
  !except=0
  heap_gc

  output_directory = "/safepool/rbyrne/fhd_outputs"
  version = "rlb_debug_uv_density_sims_May2023_2"
  vis_file_list = "/safepool/rbyrne/uv_density_simulations/calibration_error_sim/sim_uv_spacing_10_short_bls_cal_error.uvfits"

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
