pro remove_catalog_nans

  restore, "/Users/ruby/Astro/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  print, n_elements(catalog)
  use_catalog_inds = where(finite(catalog.flux.I))
  catalog = catalog[use_catalog_inds]
  print, n_elements(catalog)
  save, catalog, filename = "/Users/ruby/Astro/rlb_LWA/GLEAM_v2_plus_rlb2019_nonans.sav"

end