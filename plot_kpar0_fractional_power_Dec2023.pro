pro plot_kpar0_fractional_power_Dec2023

  plot_save_dir = '/home/rbyrne/kpar0_plots_Dec2023'

  dirty_filename = "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_image_Dec2023/ps/data/1d_binning/20230819_093023_73MHz_calibrated__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave"
  filenames = [$
  "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_image_Dec2023/ps/data/1d_binning/20230819_093023_73MHz_data_minus_cyg_cas__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",$
  "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_image_Dec2023/ps/data/1d_binning/20230819_093023_73MHz_data_minus_deGasperin_cyg_cas__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",$
  "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_image_Dec2023/ps/data/1d_binning/20230819_093023_73MHz_data_minus_deGasperin_sources__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",$
  "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_image_Dec2023/ps/data/1d_binning/20230819_093023_73MHz_data_minus_VLSS__gridded_uvf_noimgclip_dirty_xx_dft_averemove_swbh_dencorr_k0power.idlsave",$
  "/safepool/rbyrne/fhd_outputs/fhd_rlb_LWA_model_diffuse_Dec2023/ps/data/1d_binning/20230819_093023_73MHz_calibrated__gridded_uvf_noimgclip_res_xx_dft_averemove_swbh_dencorr_k0power.idlsave"$
  ]
  legend_labels = ['Cyg & Cas simple', 'de Gasperin Cyg & Cas', 'de Gasperin A Team', 'VLSSr', 'm-mode + Cyg & Cas']

  yrange = [0,100]
  xrange=[.5e-3, 3.5e-2]
  background_shading = 0

  colors = ['Navy', 'YGB4', 'Dark Green', 'ORG4', 'Crimson'] ;Choose colors with cgPickColorName()
  linestyles = [0,0,0,0,0]
  linewidths = [3,3,3,3,3]
  
  cgps_open, plot_save_dir+'/frac_power_recovered.png'
  cgDisplay, 900, 650
  
  dirty_power = getvar_savefile(dirty_filename, 'power')
  for version_ind = 0,n_elements(filenames)-1 do begin

    res_file = filenames[version_ind]
    k_edges = getvar_savefile(res_file, 'k_edges')
    power = getvar_savefile(res_file, 'power')
    frac_power = (1-sqrt(power/dirty_power))*100.

    plot_x = []
    plot_y = []
    for datapoint = 0,n_elements(frac_power)-1 do begin
      plot_y = [plot_y, frac_power[datapoint], frac_power[datapoint]]
      plot_x = [plot_x, k_edges[datapoint], k_edges[datapoint+1]]
    endfor
    if version_ind eq 0 then begin
      cgplot, plot_x, plot_y, /xlog, yrange=yrange, xrange=xrange, $
        linestyle=linestyles[version_ind], color=colors[version_ind], thick=linewidths[version_ind], title='', Charsize=1.5,$
        ytitle=textoidl('Fraction of signal modeled (%)'), xtitle=textoidl('k-perpendicular (!8h!X Mpc^{-1})'), $
        xstyle=4, /nodata, Position=[0.1, 0.22, 0.97, 0.9]
      if keyword_set(background_shading) then begin
        bl_range = [6.1, 50.]
        cgcolorfill, [xrange[0], bl_range[0]*1e-3, bl_range[0]*1e-3, xrange[0]], [yrange[0], yrange[0], yrange[1], yrange[1]], $
          color='BLK2'
        cgcolorfill, [xrange[1], bl_range[1]*1e-3, bl_range[1]*1e-3, xrange[1]], [yrange[0], yrange[0], yrange[1], yrange[1]], $
          color='BLK2'
      endif
    endif
    cgplot, plot_x, plot_y, /xlog, yrange=yrange, xrange=xrange, $
      linestyle=linestyles[version_ind], color=colors[version_ind], thick=linewidths[version_ind], /overplot, title='', Charsize=1.5,$
      ytitle=textoidl('Fraction of signal modeled (%)'), xtitle=textoidl('k-perpendicular (!8h!X Mpc^{-1})'), $
      xstyle=4 ;suppress horizontal axes
  endfor
  ; Draw lower x axis
  tick_angles_to_label = [0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 60]
  tick_angles = [.1*findgen(9, start=1), indgen(9, start=1), 10*indgen(9, start=1), 100, 180]
  tick_angles = reverse(tick_angles)
  tick_names = []
  for tick_ind=0,n_elements(tick_angles)-1 do begin
    null = where(tick_angles_to_label eq tick_angles[tick_ind], count)
    if count eq 0 then begin
      tick_names = [tick_names, ' ']
    endif else begin
      if tick_angles[tick_ind] ge 1 then use_string=strtrim(fix(tick_angles[tick_ind]), 1) $
      else use_string=STRING(tick_angles[tick_ind], FORMAT='(F3.1)')
      tick_names = [tick_names, use_string]
    endelse
  endfor
  tick_pos = 1/sin(tick_angles/180.*!Pi)
  cgAxis, 0.1, 0.1, /normal, xAxis=0, /Save, Color='black', Title='Angular scale ('+cgsymbol('deg')+')', xRange=xrange*1.e3, xstyle=1, Charsize=1.5, xlog=1,$
    xtickv=tick_pos, xticks=n_elements(tick_angles), xtickname=tick_names

  cgAxis, xaxis=1, xrange=xrange, xstyle=1, xtitle=textoidl(''), Charsize=1.5 ;draw top axis
  cgaxis, yaxis=1, yrange=yrange, ystyle=1, ytitle=textoidl(''), Charsize=1.5, yTICKFORMAT="(A1)" ;draw right axis
  cgAxis, XAxis=0, XRange=xrange*1.e3, XStyle=1, xtitle=textoidl('Baseline length (wavelengths)'), Charsize=1.5 ;draw bottom axis
  xlocation = (!X.Window[1] - !X.Window[0]) / 2  + !X.Window[0]
  ylocation = !Y.Window[1] + 2.75 * (!D.Y_CH_Size / Float(!D.Y_Size))
  cgText, xlocation, ylocation+.01, textoidl('k-perpendicular (!8h!X Mpc^{-1})'), $
    /Normal, Alignment=0.5, Charsize=1.5
  cglegend, title=legend_labels, $
    linestyle=linestyles, thick=linewidths[0], $
    color=colors, length=0.03, /center_sym, location=[.5,.45], charsize=1.1, /box, background='white', vspace=1.5
  cgControl, Resize=[800,800]
  cgps_close, /png, /delete_ps, density=800


end
