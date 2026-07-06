import numpy as np
import astropy.units as units
from astropy.coordinates import Latitude, Longitude
import pyradiosky


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


def wsclean_txt_to_pyradiosky(txt_path):

    csv_file = open(txt_path, "r")
    file_contents = csv_file.readlines()
    csv_file.close()

    header = file_contents[0].split(",")
    data = file_contents[1:]
    ncomps = len(data)

    name = np.full(ncomps, "", dtype=object)
    ra_deg = np.full(ncomps, np.nan, dtype=float)
    dec_deg = np.full(ncomps, np.nan, dtype=float)
    flux_I = np.full(ncomps, np.nan, dtype=float)

    name_ind = np.where(["Name" in header_entry for header_entry in header])[0][0]
    ra_ind = np.where(["Ra" in header_entry for header_entry in header])[0][0]
    dec_ind = np.where(["Dec" in header_entry for header_entry in header])[0][0]
    flux_ind = np.where(["I" == header_entry.strip() for header_entry in header])[0][0]

    for comp_ind in range(ncomps):
        line_split = data[comp_ind].strip("\n").split(",")

        name[comp_ind] = line_split[name_ind]
        ra_deg[comp_ind] = format_ra_deg(line_split[ra_ind])
        dec_deg[comp_ind] = format_dec_deg(line_split[dec_ind])
        flux_I[comp_ind] = line_split[flux_ind]

    nfreqs = 1
    stokes = units.Quantity(np.zeros((4, nfreqs, ncomps), dtype=float), "Jy")
    stokes[0, 0, :] = flux_I * units.Jy

    catalog = pyradiosky.SkyModel(
        name=name,
        ra=Longitude(ra_deg, units.deg),
        dec=Latitude(dec_deg, units.deg),
        stokes=stokes,
        spectral_type="flat",
        frame="icrs",
    )

    return catalog


if __name__ == "__main__":
    cat = wsclean_txt_to_pyradiosky(
        "/lustre/rbyrne/2026-04-19/20260419_055641-055832_44MHz_17h_cal_calibrated-sources.txt"
    )
    cat.write_skyh5(
        "/lustre/rbyrne/2026-04-19/20260419_055641-055832_44MHz_17h_cal_calibrated-sources.skyh5"
    )
