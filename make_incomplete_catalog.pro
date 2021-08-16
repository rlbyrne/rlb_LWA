pro make_incomplete_catalog
  
  ;Look at EoR-0 source list for reference
  skymodel = getvar_savefile('/Users/ruby/Astro/fhd_rlb_model_GLEAM_Aug2021/output_data/1061316296_skymodel.sav', 'skymodel')
  source_fluxes = skymodel.source_list.flux.I
  source_fluxes = source_fluxes[sort(source_fluxes)]
  include_flux_frac = 0.9
  
  flux_sum = 0
  source_ind = 0
  while flux_sum/total(source_fluxes) lt (1.-include_flux_frac) do begin
    flux_sum += source_fluxes[source_ind]
    source_ind += 1
  endwhile
  flux_threshold = source_fluxes[source_ind-1]
  ; Flux density threshold is 0.0913 Jy:
  print, 'Flux density threshold: '+string(flux_threshold)
  
  catalog = getvar_savefile('/Users/ruby/Astro/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav', 'catalog', /compatibility_mode)
  use_inds = where(catalog.flux.I gt flux_threshold)
  catalog = catalog[use_inds]
  save, catalog, filename='/Users/ruby/Astro/rlb_LWA/GLEAM_bright_sources.sav'

end