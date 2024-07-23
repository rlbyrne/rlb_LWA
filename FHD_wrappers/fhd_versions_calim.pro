pro fhd_versions_calim
  except=!except
  !except=0
  heap_gc

  args = Command_Line_Args(count=nargs)
  output_directory = args[0]
  version = args[1]
  vis_file_list = args[2]

  case version of

    'rlb_image_LWA_May2024': begin
        recalculate_all = 1
        instrument = 'lwa'
        import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
        calibrate_visibilities = 0
        return_cal_visibilities = 0
        model_visibilities = 0
        n_pol = 4
        image_filter_fn = "filter_uv_natural"
        split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
        save_uvf = 1
        beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_image_LWA_v2_branch_May2024': begin
        recalculate_all = 1
        instrument = 'lwa'
        import_pyuvdata_beam_filepath = '/home/rbyrne/rlb_LWA/LWAbeam_2015.fits'
        calibrate_visibilities = 0
        return_cal_visibilities = 0
        model_visibilities = 0
        n_pol = 4
        image_filter_fn = "filter_uv_natural"
        split_ps_export = 0  ;do not attempt even-odd splitting, required when only one time step is present
        save_uvf = 1
        beam_nfreq_avg = 1  ;do not average beam
    end

    'rlb_LWA_model_diffuse_Jul2024': begin
      recalculate_all = 1
      instrument = 'lwa'
      import_pyuvdata_beam_filepath = '/lustre/rbyrne/LWA_10to100_MROsoil_efields.fits'
      calibrate_visibilities = 0
      return_cal_visibilities = 0
      model_visibilities = 1
      diffuse_model = "/fast/rbyrne/skymodels/ovro_lwa_sky_map_46.992MHz_nside512.skyh5"
      diffuse_units_kelvin = 1
      model_catalog_file_path = '/fast/rbyrne/skymodels/Gasperin2020_sources_plus_48MHz.skyh5'
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
