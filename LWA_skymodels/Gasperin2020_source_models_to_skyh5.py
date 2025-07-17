import numpy as np
import healpy as hp
import pyuvdata
import pyradiosky
import pandas as pd
import sys
import astropy.units as units
from astropy.units import Quantity
from astropy.coordinates import Latitude, Longitude


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


def concatenate_catalogs(catalog_list):

    if len(catalog_list) > 1:
        for cat_ind in range(len(catalog_list) - 1):
            if cat_ind == 0:
                cat1 = catalog_list[cat_ind]
            cat2 = catalog_list[cat_ind + 1]
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
                spectral_index=np.concatenate(
                    (cat1.spectral_index, cat2.spectral_index)
                ),
                frame="icrs",
            )
            catalog.check()
            cat1 = catalog

    return catalog


def create_deGasperin_catalog(use_freq, source_file_dir="/lustre/rbyrne/skymodels"):

    cas_cat = convert_wsclean_txt_models_to_pyradiosky(
        f"{source_file_dir}/Cas-sources.txt",
        use_freq,
        source_name="Cas",
    )
    cyg_cat = convert_wsclean_txt_models_to_pyradiosky(
        f"{source_file_dir}/Cyg-sources.txt",
        use_freq,
        source_name="Cyg",
    )
    tau_cat = convert_wsclean_txt_models_to_pyradiosky(
        f"{source_file_dir}/Tau-sources.txt",
        use_freq,
        source_name="Tau",
    )
    vir_cat = convert_wsclean_txt_models_to_pyradiosky(
        f"{source_file_dir}/Vir-sources.txt",
        use_freq,
        source_name="Vir",
    )

    combined_cat = concatenate_catalogs([cas_cat, cyg_cat, tau_cat, vir_cat])
    return combined_cat


def get_supplementary_source_info():
    # This script gets data for additional sources that were peeled from the Eastwood et al.
    # m-mode map.
    # Source spectral information is from Perley & Butler 2017
    # Source locations are from https://ned.ipac.caltech.edu/

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


def interpolate_flux_supplementary_sources(spectral_coeffs, target_freq_hz):

    log_flux = 0
    for coeff_ind, coeff in enumerate(spectral_coeffs):
        log_flux += coeff * np.log(target_freq_hz / 1e9) ** coeff_ind
    return np.exp(log_flux)


def interpolate_spec_ind_supplementary_sources(spectral_coeffs, target_freq_hz):

    spec_ind = 0
    for coeff_ind, coeff in enumerate(spectral_coeffs[1:]):
        spec_ind += coeff * (coeff_ind + 1) * np.log(target_freq_hz / 1e9) ** coeff_ind
    return spec_ind


def create_skymodel_supplementary_sources(target_freq_hz):

    source_dict = get_supplementary_source_info()
    names = list(source_dict.keys())
    ras = []
    decs = []
    stokes = Quantity(np.zeros((4, 1, len(names)), dtype=float), "Jy")
    spec_inds = []
    for source_ind, name in enumerate(names):
        ras.append(source_dict[name]["ra"])
        decs.append(source_dict[name]["dec"])
        flux = interpolate_flux_supplementary_sources(
            source_dict[name]["spectral_coeffs"], target_freq_hz
        )
        stokes[0, 0, source_ind] = Quantity(flux, "Jy")
        spectral_index = interpolate_spec_ind_supplementary_sources(
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


def create_combined_catalog(target_freq_hz):

    deGasperin_sources = create_deGasperin_catalog(target_freq_hz)
    supplementary_sources = create_skymodel_supplementary_sources(target_freq_hz)
    combined_catalog = concatenate_catalogs([deGasperin_sources, supplementary_sources])
    return combined_catalog


def create_point_source_catalog(
    input_catalog,
):  # Convert a catalog with extended sources into one with point sources only

    sources = [
        source_name
        for source_name in list(set(input_catalog.extended_model_group))
        if len(source_name) > 0
    ]
    catalog_list = []
    for source_name in sources:
        source_inds = np.where(input_catalog.extended_model_group == source_name)[0]
        if len(source_inds) > 0:
            component_flux = input_catalog.stokes[0, 0, source_inds]
            weighted_mean_ra = np.sum(
                component_flux.value * input_catalog.ra[source_inds]
            ) / np.sum(component_flux.value)
            weighted_mean_dec = np.sum(
                component_flux.value * input_catalog.dec[source_inds]
            ) / np.sum(component_flux.value)
            weighted_mean_spectral_index = np.sum(
                component_flux.value * input_catalog.spectral_index[source_inds]
            ) / np.sum(component_flux.value)
            catalog = pyradiosky.SkyModel(
                name=input_catalog.name[source_inds[0]],
                extended_model_group=source_name,
                ra=Longitude(weighted_mean_ra),
                dec=Latitude(weighted_mean_dec),
                stokes=np.sum(input_catalog.stokes[:, :, source_inds], axis=2),
                spectral_type="spectral_index",
                reference_frequency=np.mean(
                    input_catalog.reference_frequency[source_inds]
                ),
                spectral_index=np.array([weighted_mean_spectral_index]),
                frame="icrs",
            )
            catalog_list.append(catalog)
    output_catalog = concatenate_catalogs(catalog_list)
    return output_catalog


def create_catalog_from_Gregg_source_fluxes(
    freq_array,
    flux_csv="/lustre/rbyrne/20250519_LST_232293p96_model_summary_intrinsic_only.csv",
):

    deGasperin_sources = create_deGasperin_catalog(np.mean(freq_array))
    point_source_catalog = create_point_source_catalog(deGasperin_sources)
    keep_comp_inds = [
        ind
        for ind in range(point_source_catalog.Ncomponents)
        if point_source_catalog.name[ind][:3] in ["Cas", "Cyg", "Vir"]
    ]
    point_source_catalog.select(component_inds=keep_comp_inds)
    point_source_catalog.spectral_type = "full"
    point_source_catalog.Nfreqs = len(freq_array)
    point_source_catalog.freq_array = freq_array
    point_source_catalog.reference_frequency = None
    point_source_catalog.stokes = np.zeros(
        (4, point_source_catalog.Nfreqs, point_source_catalog.Ncomponents), dtype=float
    )
    point_source_catalog = pyradiosky.SkyModel(
        name=point_source_catalog.name,
        ra=point_source_catalog.ra,
        dec=point_source_catalog.dec,
        stokes=Quantity(np.zeros(
            (4, len(freq_array), point_source_catalog.Ncomponents), dtype=float), "Jy"
        ),
        spectral_type="full",
        freq_array=Quantity(freq_array, "hertz"),
        frame="icrs",
    )
    print(point_source_catalog.freq_array)

    csv_data = pd.read_csv(flux_csv)
    frequencies_orig = csv_data["# Frequency_Hz"]
    for source_ind in range(point_source_catalog.Ncomponents):
        use_key = [
            key
            for key in csv_data.keys()
            if key.startswith(point_source_catalog.name[source_ind][:3])
        ][0]
        flux_vals = csv_data[use_key]
        polyfit_vals = np.polyfit(frequencies_orig, flux_vals, 20)
        smoothed_vals = np.polyval(
            polyfit_vals,
            freq_array,
        )
        point_source_catalog.stokes[0, :, source_ind] = Quantity(smoothed_vals, "Jy")
    point_source_catalog.check()
    return point_source_catalog


if __name__ == "__main__":

    if False:
        for file_name in [
            "18",
            "23",
            "27",
            "36",
            "41",
            "46",
            "50",
            "55",
            "59",
            "64",
            "73",
            "78",
            "82",
        ]:

            input_file = f"/lustre/gh/2024-03-02/calibration/ruby/{file_name}.ms"
            uv = pyuvdata.UVData()
            uv.read(input_file)
            use_freq = np.mean(uv.freq_array)
            print(use_freq / 1e6)
            catalog = create_combined_catalog(use_freq)
            catalog.write_skyh5(
                f"/fast/rbyrne/skymodels/Gasperin2020_sources_plus_{file_name}.skyh5",
                clobber=True,
            )
            point_source_catalog = create_point_source_catalog(catalog)
            point_source_catalog.write_skyh5(
                f"/fast/rbyrne/skymodels/Gasperin2020_point_sources_plus_{file_name}.skyh5",
                clobber=True,
            )
    freq_array = np.arange(0, 87e6, 23925.78125)
    freq_array = freq_array[np.where(freq_array > 13e6)]
    print(freq_array)
    cat = create_catalog_from_Gregg_source_fluxes(
        freq_array,
    )
    cat.write_skyh5("/lustre/rbyrne/skymodels/Gregg_20250519_source_models.skyh5", clobber=True)
