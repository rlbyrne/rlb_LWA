pro ps_single_obs_wrapper

  args = Command_Line_Args(count=nargs)
  obs_id=args[0]
  outdir=args[1]
  fhd_version=args[2]
  refresh_ps=args[3]
  uvf_input=args[4]
  no_evenodd=args[5]

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

  if refresh_ps eq 0 then undefine, refresh_ps
  if uvf_input eq 0 then undefine, uvf_input
  if no_evenodd eq 0 then undefine, no_evenodd

  ps_wrapper, outdir+'/fhd_'+fhd_version, obs_id, /png, /plot_kpar_power, $
    refresh_ps=refresh_ps, uvf_input=uvf_input, pol_inc=["xx"], $
    no_evenodd=no_evenodd, refresh_info=refresh_ps


end
