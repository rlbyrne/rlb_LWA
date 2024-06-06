import pyradiosky
import numpy as np
import astropy.units as units
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude

# Script for debugging skymodel concatenation issues


def format_ra_deg(ra_str):

    ra_split = ra_str.split(":")
    ra_val = (
        np.abs(int(ra_split[0]))
        + int(ra_split[1]) / 60.0
        + float(ra_split[2]) / 60.0**2.0
    )
    if "-" in ra_str:
        ra_val *= -1
    ra_val *= 15.0  # Convert from hours to degrees
    return ra_val


def format_dec_deg(dec_str):

    dec_split = dec_str.split(".")
    dec_val = (
        np.abs(int(dec_split[0]))
        + int(dec_split[1]) / 60.0
        + int(dec_split[2]) / 60.0**2.0
    )
    if len(dec_split) > 3:
        dec_val += float(f"0.{dec_split[3]}") / 60.0**2.0
    if "-" in dec_str:
        dec_val *= -1
    return dec_val


def convert_wsclean_txt_models_to_pyradiosky(txt_path, target_freq_hz, source_name=""):

    file = open(txt_path, "r")
    file_contents = file.readlines()
    file.close()

    header = file_contents[0]
    data = file_contents[1:]
    ncomps = len(data)

    name = np.full(ncomps, "", dtype=object)
    extended_model_group = np.full(ncomps, source_name, dtype=object)
    ra_deg = np.full(ncomps, np.nan, dtype=float)
    dec_deg = np.full(ncomps, np.nan, dtype=float)
    flux_I_orig = np.full(ncomps, np.nan, dtype=float)

    for comp_ind in range(ncomps):
        line_split = data[comp_ind].strip("\n").split(",")
        name[comp_ind] = f"{source_name}_{line_split[0]}"
        ra_deg[comp_ind] = format_ra_deg(line_split[2])
        dec_deg[comp_ind] = format_dec_deg(line_split[3])
        flux_I_orig[comp_ind] = line_split[4]

    nfreqs = 1
    stokes = Quantity(np.zeros((4, nfreqs, ncomps), dtype=float), "Jy")
    stokes[0, 0, :] = flux_I_orig * units.Jy

    catalog = pyradiosky.SkyModel(
        name=name,
        extended_model_group=extended_model_group,
        ra=Longitude(ra_deg, units.deg),
        dec=Latitude(dec_deg, units.deg),
        stokes=stokes,
        spectral_type="spectral_index",
        reference_frequency=Quantity(np.full(ncomps, target_freq_hz), "hertz"),
        spectral_index=np.full(
            ncomps, -0.8
        ),  # Spoof spectral index because convention does not match
        frame="icrs",
    )

    return catalog


if __name__ == "__main__":

    use_freq = 57e6
    cas_cat = convert_wsclean_txt_models_to_pyradiosky(
        "/Users/ruby/Astro/Gasperin2020_source_models/Cas-sources.txt",
        use_freq,
        source_name="Cas",
    )
    cyg_cat = convert_wsclean_txt_models_to_pyradiosky(
        "/Users/ruby/Astro/Gasperin2020_source_models/Cyg-sources.txt",
        use_freq,
        source_name="Cyg",
    )
    tau_cat = convert_wsclean_txt_models_to_pyradiosky(
        "/Users/ruby/Astro/Gasperin2020_source_models/Tau-sources.txt",
        use_freq,
        source_name="Tau",
    )

    cas_cyg = cas_cat.concat(cyg_cat, inplace=False)
    print(cas_cyg.skycoord.representation_type)
    print(tau_cat.skycoord.representation_type)
    combined_cat = cas_cyg.concat(tau_cat, inplace=False)
