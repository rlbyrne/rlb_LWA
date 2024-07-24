import numpy as np
import pyradiosky
import sys
import astropy.units as units
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude

# This script has been deprecated and combined with Gasperin2020_source_models_to_skyh5.py
# This script gets data for additional sources that were peeled from the Eastwood et al.
# m-mode map.
# Source spectral information is from Perley & Butler 2017
# Source locations are from https://ned.ipac.caltech.edu/


def get_source_info():

    source_dict = {"Her A": {}, "Hya A": {}, "3C 123": {}, "3C 353": {}}

    ## Her A (Hercules A, 3C 348)
    source_dict["Her A"] = {
        "spectral_coeffs": [1.8298, -1.0247, -0.0951],
        "ra": Longitude((16.0 + 51.0 / 60.0 + 8.1468 / 60.0**2.0) * 15.0, units.deg),
        "dec": Latitude((4.0 + 59.0 / 60.0 + 33.316 / 60.0**2.0), units.deg),
    }

    ## Hya A (Hydra A, 3C 218)
    source_dict["Hya A"] = {
        "spectral_coeffs": [
            1.7795,
            -0.9176,
            -0.0843,
            -0.0139,
            0.0295,
        ],
        "ra": Longitude((9.0 + 18.0 / 60.0 + 5.6687 / 60.0**2.0) * 15.0, units.deg),
        "dec": Latitude((-12.0 - 5.0 / 60.0 - 43.953 / 60.0**2.0), units.deg),
    }

    ## 3C 123
    source_dict["3C 123"] = {
        "spectral_coeffs": [
            1.8017,
            -0.7884,
            -0.1035,
            -0.0248,
            0.0090,
        ],
        "ra": Longitude((4.0 + 37.0 / 60.0 + 4.3752 / 60.0**2.0) * 15.0, units.deg),
        "dec": Latitude((29.0 + 40.0 / 60.0 + 13.818 / 60.0**2.0), units.deg),
    }

    ## 3C 353
    source_dict["3C 353"] = {
        "spectral_coeffs": [1.8627, -0.6938, -0.0998, -0.0732],
        "ra": Longitude((17.0 + 20.0 / 60.0 + 28.1582 / 60.0**2.0) * 15.0, units.deg),
        "dec": Latitude((0.0 - 58.0 / 60.0 - 46.621 / 60.0**2.0), units.deg),
    }

    return source_dict


def interpolate_flux(spectral_coeffs, target_freq_hz):

    log_flux = 0
    for coeff_ind, coeff in enumerate(spectral_coeffs):
        log_flux += coeff * np.log(target_freq_hz / 1e9) ** coeff_ind
    return np.exp(log_flux)


def interpolate_spec_ind(spectral_coeffs, target_freq_hz):

    spec_ind = 0
    for coeff_ind, coeff in enumerate(spectral_coeffs[1:]):
        spec_ind += coeff * (coeff_ind + 1) * np.log(target_freq_hz / 1e9) ** coeff_ind
    return spec_ind


def create_skymodel(target_freq_hz):

    source_dict = get_source_info()
    names = list(source_dict.keys())
    ras = []
    decs = []
    stokes = Quantity(np.zeros((4, 1, len(names)), dtype=float), "Jy")
    spec_inds = []
    for source_ind, name in enumerate(names):
        ras.append(source_dict[name]["ra"])
        decs.append(source_dict[name]["dec"])
        flux = interpolate_flux(source_dict[name]["spectral_coeffs"], target_freq_hz)
        stokes[0, 0, source_ind] = Quantity(flux, "Jy")
        spectral_index = interpolate_spec_ind(
            source_dict[name]["spectral_coeffs"], target_freq_hz
        )
        spec_inds.append(spectral_index)

    catalog = pyradiosky.SkyModel(
        name=names,
        extended_model_group=names,
        ra=ras,
        dec=decs,
        stokes=stokes,
        spectral_type="spectral_index",
        reference_frequency=Quantity(np.full(len(names), target_freq_hz), "hertz"),
        spectral_index=spec_inds,
        frame="icrs",
    )
    if not catalog.check():
        print("WARNING: Catalog check failed.")

    return catalog


if __name__ == "__main__":
    catalog = create_skymodel(47839599.609375)

    degasperin_catalog = pyradiosky.SkyModel()
    degasperin_catalog.read(
        "/Users/ruby/Astro/Gasperin2020_source_models/Gasperin2020_sources_48MHz.skyh5"
    )

    catalog.concat(degasperin_catalog, inplace=True)
    catalog.write_skyh5(
        "/Users/ruby/Astro/Gasperin2020_source_models/Gasperin2020_sources_plus_48MHz.skyh5",
        clobber=True,
    )
