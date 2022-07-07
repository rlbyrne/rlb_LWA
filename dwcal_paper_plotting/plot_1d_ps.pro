pro plot_1d_ps

  yrange = [1.e1, 1.e8]
  data_path = "/Users/ruby/Astro/fhd_outputs/fhd_rlb_cal_sims_Jun2022/data/1d_binning"
  data_filenames = ["unity_gains_gridded_uvf_even_odd_joint_noimgclip__diagonal_minus_uncalib__res_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave",$
    "unity_gains_gridded_uvf_even_odd_joint_noimgclip__dwcal_minus_uncalib__res_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave",$
    "gains_gridded_uvf_even_odd_joint_noimgclip__randomdiagonal_minus_unityuncalib__res_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave",$
    "gains_gridded_uvf_even_odd_joint_noimgclip__randomdwcal_minus_unityuncalib__res_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave",$
    "gains_gridded_uvf_even_odd_joint_noimgclip__ripplediagonal_minus_unityuncalib__res_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave",$
    "gains_gridded_uvf_even_odd_joint_noimgclip__rippledwcal_minus_unityuncalib__res_xx_noimgclip_dft_averemove_swbh_dencorr_no_horizon_wedge_1dkpower.idlsave"]
  datafiles = []
  for file_ind=0,n_elements(data_filenames)-1 do datafiles=[datafiles, data_path+"/"+data_filenames[file_ind]]
    
  colors = ['cyan', 'salmon', 'dodger blue', 'red', 'Blu8', 'red7']
  linestyles = [0,0,2,2,1,1]
  linewidths = [4,4,4,4,5,5]
  
  eor_file = '/Users/ruby/Astro/FHD/catalog_data/simulation/eor_power_1d.idlsave'
  eor_k_centers = getvar_savefile(eor_file, 'k_centers')
  eor_power = getvar_savefile(eor_file, 'power')
  eor_plot_x = []
  eor_plot_y = []
  for datapoint = 0,n_elements(eor_power)-1 do begin
    eor_plot_y = [eor_plot_y, eor_power[datapoint], eor_power[datapoint]]
    if (datapoint ne 0) and (datapoint ne n_elements(eor_power)-1) then begin
      eor_plot_x = [eor_plot_x, (eor_k_centers[datapoint]+eor_k_centers[datapoint-1])/2, (eor_k_centers[datapoint]+eor_k_centers[datapoint+1])/2]
    endif else begin
      if datapoint eq 0 then eor_plot_x = [eor_plot_x, eor_k_centers[datapoint], (eor_k_centers[datapoint]+eor_k_centers[datapoint+1])/2]
      if datapoint eq n_elements(eor_power)-1 then eor_plot_x = [eor_plot_x, (eor_k_centers[datapoint]+eor_k_centers[datapoint-1])/2, eor_k_centers[datapoint]]
    endelse
  endfor
      
  cgps_open, '/Users/ruby/Astro/dwcal_paper_plots/1d_ps.png'
  cgplot, eor_plot_x, eor_plot_y, /ylog, yrange=yrange, xrange=[.1, 1.2], $
    linestyle=0, color='black', thick=3, $
    ytitle=textoidl('P_k (mK^2 !8h!X^{-3} Mpc^3)'), xtitle=textoidl('k (!8h!X Mpc^{-1})')
  for file_ind = 0,n_elements(datafiles)-1 do begin
    k_edges = getvar_savefile(datafiles[file_ind], 'k_edges')
    power = getvar_savefile(datafiles[file_ind], 'power')
    power[where(power lt yrange[0]/100.)] = yrange[0]/100.  ; Remove negative values for log plot
    
    print, mean(power)

    plot_x = []
    plot_y = []
    for datapoint = 0,n_elements(power)-1 do begin
      plot_y = [plot_y, power[datapoint], power[datapoint]]
      plot_x = [plot_x, k_edges[datapoint], k_edges[datapoint+1]]
    endfor
    cgplot, plot_x, plot_y, /ylog,$
      linestyle=linestyles[file_ind], color=colors[file_ind], thick=linewidths[file_ind], /overplot
  endfor
  cglegend, title=['Sky-Based Cal Error', 'DWCal Error', 'Predicted EoR Signal'], $
    linestyle=[0,0,0], thick=4, $
    color=['blue', 'red', 'black'], length=0.03, /center_sym, location=[.6,.85], charsize=.8, /box, background='white'
  cgps_close, /png, /delete_ps, density=1000
    

end