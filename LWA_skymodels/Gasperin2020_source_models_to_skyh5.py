import numpy as np
import healpy as hp
import scipy
import pyradiosky
import sys
import astropy.units as units
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude
import pyradiosky_utils


# Source models downloaded from http://cdsarc.u-strasbg.fr/viz-bin/cat/J/A+A/635/A150#/browse
# Reference: de Gasperin et al. 2020, https://doi.org/10.1051/0004-6361/201936844
# For more information about the model format, see https://sourceforge.net/p/wsclean/wiki/ComponentList/


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


def interpolate_flux(
    flux_I_orig, freq_orig, freq_new, spectral_indices, use_log_spectral_ind
):
    # See https://sourceforge.net/p/wsclean/wiki/ComponentList/ for reference

    flux_I_new = np.copy(flux_I_orig)

    log_indices = np.where(use_log_spectral_ind)[0]
    if len(log_indices) > 0:
        for ind in range(np.shape(spectral_indices)[1]):
            flux_I_new[log_indices] += np.exp(
                spectral_indices[log_indices, ind]
                * np.log(freq_new / freq_orig[log_indices]) ** (ind + 1)
            )

    ordinary_indices = np.where(~use_log_spectral_ind)[0]
    if len(ordinary_indices) > 0:
        for ind in range(np.shape(spectral_indices)[1]):
            flux_I_new[ordinary_indices] += spectral_indices[ordinary_indices, ind] * (
                freq_new / freq_orig[ordinary_indices] - 1
            ) ** (ind + 1)

    return flux_I_new


def fit_spectral_index(
    flux_I_new, freq_orig, freq_new, spectral_indices, use_log_spectral_ind
):

    spectral_indices_fit = np.zeros((np.shape(spectral_indices)[0]), dtype=float)

    log_indices = np.where(use_log_spectral_ind)[0]
    if len(log_indices) > 0:
        sum_term = np.zeros((len(log_indices)), dtype=float)
        for ind in range(np.shape(spectral_indices)[1]):
            sum_term += (
                spectral_indices[log_indices, ind]
                * (ind + 1)
                * np.log(freq_new / freq_orig[log_indices]) ** ind
            )
        spectral_indices_fit[log_indices] = sum_term

    ordinary_indices = np.where(~use_log_spectral_ind)[0]
    if len(ordinary_indices) > 0:
        sum_term = np.zeros((len(ordinary_indices)), dtype=float)
        for ind in range(np.shape(spectral_indices)[1]):
            sum_term += (
                1
                / freq_orig[ordinary_indices]
                * spectral_indices[ordinary_indices, ind]
                * (ind + 1)
                * (freq_new / freq_orig[ordinary_indices] - 1) ** ind
            )
        spectral_indices_fit[ordinary_indices] = (
            freq_new / flux_I_new[ordinary_indices] * sum_term
        )

    return spectral_indices_fit


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
    spectral_indices = np.zeros((ncomps, 6), dtype=float)
    use_log_spectral_ind = np.zeros((ncomps), dtype=bool)
    freq_hz = np.full(ncomps, np.nan, dtype=float)
    major_axis_arcsec = np.zeros(ncomps, dtype=float)  # Not used by pyradiosky
    minor_axis_arcsec = np.zeros(ncomps, dtype=float)  # Not used by pyradiosky
    orientation = np.full(ncomps, np.nan, dtype=float)  # Not used by pyradiosky

    for comp_ind in range(ncomps):
        line_split = data[comp_ind].strip("\n").split(",")

        # Deal with bracketed entries with more than one sub-entry
        line_split_new = []
        ind = 0
        while ind < len(line_split):
            if "[" in line_split[ind]:
                bracket_chunk = []
                while "]" not in line_split[ind]:
                    bracket_chunk.append(line_split[ind].strip("[").strip("]"))
                    ind += 1
                bracket_chunk.append(line_split[ind].strip("[").strip("]"))
                line_split_new.append(bracket_chunk)
                ind += 1
            else:
                line_split_new.append(line_split[ind])
                ind += 1

        name[comp_ind] = f"{source_name}_{line_split_new[0]}"
        ra_deg[comp_ind] = format_ra_deg(line_split_new[2])
        dec_deg[comp_ind] = format_dec_deg(line_split_new[3])
        flux_I_orig[comp_ind] = line_split_new[4]
        if isinstance(line_split_new[5], list):
            for ind, spec_ind_value in enumerate(line_split_new[5]):
                spectral_indices[comp_ind, ind] = spec_ind_value
        else:
            spectral_indices[comp_ind, 0] = line_split_new[5]
        if line_split_new[6] == "false":
            use_log_spectral_ind[comp_ind] = False
        elif line_split_new[6] == "true":
            use_log_spectral_ind[comp_ind] = True
        else:
            print("WARNING: Unknown spectral index type.")
        freq_hz[comp_ind] = line_split_new[7]
        if len(line_split_new[8]) > 0:
            major_axis_arcsec[comp_ind] = line_split_new[8]
        if len(line_split_new[9]) > 0:
            minor_axis_arcsec[comp_ind] = line_split_new[9]
        if len(line_split_new[10]) > 0:
            orientation[comp_ind] = line_split_new[10]

    # Convert flux values to target freq
    flux_I = interpolate_flux(
        flux_I_orig, freq_hz, target_freq_hz, spectral_indices, use_log_spectral_ind
    )
    # Interpolate spectral indices to a single number. Use the same spectral index for all components.
    if np.min(use_log_spectral_ind):
        print(
            "ERROR: Not all spectral indices are use the ordinary polynomial convention. Exiting."
        )
        sys.exit(1)
    spectral_indices_use = fit_spectral_index(
        np.array([np.sum(flux_I)]),
        np.array([np.mean(freq_hz)]),
        target_freq_hz,
        np.sum(spectral_indices, axis=0)[np.newaxis, :],
        np.array([False]),
    )
    spectral_indices_use = np.full(ncomps, spectral_indices_use)

    nfreqs = 1
    stokes = Quantity(np.zeros((4, nfreqs, ncomps), dtype=float), "Jy")
    stokes[0, 0, :] = flux_I * units.Jy

    catalog = pyradiosky.SkyModel(
        name=name,
        extended_model_group=extended_model_group,
        ra=Longitude(ra_deg, units.deg),
        dec=Latitude(dec_deg, units.deg),
        stokes=stokes,
        spectral_type="spectral_index",
        reference_frequency=Quantity(np.full(ncomps, target_freq_hz), "hertz"),
        spectral_index=spectral_indices_use,
        frame="icrs",
    )

    return catalog


def concatenate_catalogs(cat1, cat2):

    catalog = pyradiosky.SkyModel(
        name=np.concatenate((cat1.name, cat2.name)),
        extended_model_group=np.concatenate(
            (cat1.extended_model_group, cat2.extended_model_group)
        ),
        ra=np.concatenate((cat1.ra, cat2.ra)),
        dec=np.concatenate((cat1.dec, cat2.dec)),
        stokes=np.concatenate((cat1.stokes, cat2.stokes), axis=2),
        spectral_type="spectral_index",
        reference_frequency=np.concatenate(
            (cat1.reference_frequency, cat2.reference_frequency)
        ),
        spectral_index=np.concatenate((cat1.spectral_index, cat2.spectral_index)),
        frame="icrs",
    )
    catalog.check()
    return catalog


if __name__ == "__main__":

    use_freq = 47839599.609375
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
    vir_cat = convert_wsclean_txt_models_to_pyradiosky(
        "/Users/ruby/Astro/Gasperin2020_source_models/Vir-sources.txt",
        use_freq,
        source_name="Vir",
    )

    cas_cyg = concatenate_catalogs(cas_cat, cyg_cat)
    combined_cat = concatenate_catalogs(cas_cyg, tau_cat)
    combined_cat = concatenate_catalogs(combined_cat, vir_cat)
    cas_cyg.write_skyh5(
        "/Users/ruby/Astro/Gasperin2020_source_models/Gasperin2020_cyg_cas_48MHz.skyh5",
        clobber=True,
    )
    combined_cat.write_skyh5(
        "/Users/ruby/Astro/Gasperin2020_source_models/Gasperin2020_sources_48MHz.skyh5",
        clobber=True,
    )

    # Model Cas A as a single point source
    if False:
        cas_cat_single_source = pyradiosky.SkyModel(
            name=cas_cat.name[0],
            extended_model_group=cas_cat.extended_model_group[0],
            ra=np.mean(cas_cat.ra),
            dec=np.mean(cas_cat.dec),
            stokes=np.sum(cas_cat.stokes, axis=2),
            spectral_type="spectral_index",
            reference_frequency=Quantity(np.full(1, use_freq), "hertz"),
            spectral_index=0.0,
            frame="icrs",
        )
        cas_cat_single_source.write_skyh5(
            "/Users/ruby/Astro/Gasperin2020_source_models/Cas_single_source_48MHz.skyh5",
            clobber=True,
        )
    # pyradiosky_utils.plot_skymodel(combined_cat)
