import pyradiosky
import numpy as np
import scipy.io
import astropy.units as units
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude
import sys


def convert_sav_catalog_to_pyradiosky(sav_path):

    sav_catalog = scipy.io.readsav(sav_path)["catalog"]

    ras = []
    decs = []
    flux_I = []
    flux_Q = []
    flux_U = []
    flux_V = []
    freq = []
    name = []
    extended_model_group = []
    spec_ind = []

    for source_ind, source in enumerate(sav_catalog):
        if source["extend"] is None:
            ras.append(source["ra"])
            decs.append(source["dec"])
            flux_I.append(source["flux"]["I"][0])
            flux_Q.append(source["flux"]["Q"][0])
            flux_U.append(source["flux"]["U"][0])
            flux_V.append(source["flux"]["V"][0])
            freq.append(source["freq"])
            name.append(str(source_ind))
            extended_model_group.append(str(source_ind))
            spec_ind.append(source["alpha"])
        else:
            for comp_ind, comp in enumerate(source["extend"]):
                ras.append(comp["ra"])
                decs.append(comp["dec"])
                flux_I.append(comp["flux"]["I"][0])
                flux_Q.append(comp["flux"]["Q"][0])
                flux_U.append(comp["flux"]["U"][0])
                flux_V.append(comp["flux"]["V"][0])
                freq.append(comp["freq"])
                name.append(f"{source_ind}_{comp_ind}")
                extended_model_group.append(str(source_ind))
                spec_ind.append(comp["alpha"])

    nfreqs = 1
    stokes = Quantity(np.zeros((4, nfreqs, len(flux_I)), dtype=float), "Jy")
    stokes[0, 0, :] = flux_I * units.Jy
    stokes[1, 0, :] = flux_Q * units.Jy
    stokes[2, 0, :] = flux_U * units.Jy
    stokes[3, 0, :] = flux_V * units.Jy

    catalog = pyradiosky.SkyModel(
        name=name,
        extended_model_group=extended_model_group,
        ra=Longitude(ras, units.deg),
        dec=Latitude(decs, units.deg),
        stokes=stokes,
        spectral_type="spectral_index",
        reference_frequency=Quantity(np.array(freq) * 1e6, "hertz"),
        spectral_index=spec_ind,
        component_type="point",
    )
    if not catalog.check():
        print("ERROR: Catalog check failed.")
        sys.exit(1)

    return catalog


if __name__ == "__main__":

    catalog = convert_sav_catalog_to_pyradiosky(
        "/Users/ruby/Astro/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
    )
    catalog.write_skyh5("/Users/ruby/Astro/GLEAM_v2_plus_rlb2019.skyh5")
