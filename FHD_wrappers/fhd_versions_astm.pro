pro fhd_versions_astm
  except=!except
  !except=0
  heap_gc

  version = 'rlb_test_run_Jul2021'
  output_directory = '/lustre/rbyrne/fhd_outputs'

  case version of

    'rlb_test_run_Jul2021': begin
      obs_id = '1061316296'
      vis_file_list = '/lustre/rbyrne/MWA_data/'+obs_id+'.uvfits'
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
