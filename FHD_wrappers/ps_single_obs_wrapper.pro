pro ps_single_obs_wrapper

  args = Command_Line_Args(count=nargs)
  obs_id=args[0]
  outdir=args[1]
  fhd_version=args[2]
  refresh_ps=args[3]
  uvf_input=args[4]
  no_evenodd=args[5]
  xx_only=args[6]
  float_colorbar=args[7]

  refresh_ps = fix(refresh_ps)
  if (refresh_ps ne 0) and (refresh_ps ne 1) then begin
    print, 'Parameter refresh_ps must be 0 or 1. Returning.'
    return
  endif
  uvf_input = fix(uvf_input)
  if (uvf_input ne 0) and (uvf_input ne 1) then begin
    print, 'Parameter uvf_input must be 0 or 1. Returning.'
    return
  endif
  no_evenodd = fix(no_evenodd)
  if (no_evenodd ne 0) and (no_evenodd ne 1) then begin
    print, 'Parameter no_evenodd must be 0 or 1. Returning.'
    return
  endif
  xx_only = fix(xx_only)
  if (xx_only ne 0) and (xx_only ne 1) then begin
    print, 'Parameter xx_only must be 0 or 1. Returning.'
    return
  endif
  if (float_colorbar ne 0) and (float_colorbar ne 1) then begin
    print, 'Parameter float_colorbar must be 0 or 1. Returning.'
    return
  endif

  if refresh_ps eq 0 then undefine, refresh_ps
  if uvf_input eq 0 then undefine, uvf_input
  if no_evenodd eq 0 then undefine, no_evenodd
  if float_colorbar eq 1 then set_data_ranges=1 else set_data_ranges=0

  if xx_only eq 1 then begin
    ps_wrapper, outdir+'/fhd_'+fhd_version, obs_id, /png, /plot_kpar_power, $
    refresh_ps=refresh_ps, uvf_input=uvf_input, pol_inc=["xx"], $
    no_evenodd=no_evenodd, refresh_info=refresh_ps, set_data_ranges=set_data_ranges
  endif else begin
    ps_wrapper, outdir+'/fhd_'+fhd_version, obs_id, /png, /plot_kpar_power, $
    refresh_ps=refresh_ps, uvf_input=uvf_input,$
    no_evenodd=no_evenodd, refresh_info=refresh_ps, set_data_ranges=set_data_ranges
  endelse


end
