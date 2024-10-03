from astropy.io import fits
import numpy as np
import healpy as hp
import pyradiosky
import astropy.units as units
from astropy.units import Quantity
import sys


def convert_fits_to_pyradiosky(
    fits_filepath,
    freq_mhz,
    input_frame=None,  # Options are "galactic", "equatorial", or None. Overwrites file coordsys if not None.
    output_frame=None,  # Options are "galactic", "equatorial", or None. If None, input frame is preserved.
    output_nside=None,  # If None, input nside is preserved
):

    file_contents = fits.open(fits_filepath)
    map_data = np.array(file_contents[1].data, dtype=float)
    ordering = file_contents[1].header["ORDERING"].lower()
    nside = int(file_contents[1].header["NSIDE"])
    npix = hp.nside2npix(nside)
    coordsys = file_contents[1].header["COORDSYS"]
    file_contents.close()

    history_str = f"From file {fits_filepath.split('/')[-1]}"

    if input_frame is not None:
        if input_frame == "equatorial":
            use_coordsys = "C"
        elif input_frame == "galactic":
            use_coordsys = "G"
        else:
            sys.exit(
                "ERROR: Unsupported input_frame. Options are equatorial, galactic, or None."
            )
        if use_coordsys != coordsys:
            print(
                f"WARNING: Input coordinate system mismatch. Assuming {input_frame} coordinates; ignoring file contents."
            )
            coordsys = use_coordsys

    if output_frame is not None:
        # COORDSYS definitions: G = galactic, E = ecliptic, C = celestial = equatorial
        if output_frame == "equatorial":
            output_frame_code = "C"
        elif output_frame == "galactic":
            output_frame_code = "G"
        else:
            sys.exit(
                "ERROR: Unsupported output_frame. Options are equatorial, galactic, or None."
            )

        if coordsys != output_frame_code:  # Transform coordinate frame
            if ordering == "nest":
                nest = True
            elif ordering == "ring":
                nest = False
            else:
                print("WARNING: Unknown ordering. Assuming ring ordering.")
                nest = False
            theta_init, phi_init = hp.pixelfunc.pix2ang(
                nside, np.arange(npix), nest=nest
            )
            rot = hp.rotator.Rotator(coord=[output_frame_code, coordsys])
            theta_rot, phi_rot = rot(theta_init, phi_init)
            map_data = hp.get_interp_val(map_data, theta_rot, phi_rot, nest=nest)
            history_str = f"{history_str}, interpolated to {output_frame} frame"
            coordsys = output_frame_code

    if coordsys == "C":
        frame_str = "icrs"
    elif coordsys == "G":
        frame_str = "galactic"
    else:
        sys.exit("ERROR: Unsupported coordsys.")

    if output_nside is not None and output_nside != nside:  # Interpolate to new nside
        map_data = hp.pixelfunc.ud_grade(
            map_data, output_nside, pess=True, order_in=ordering
        )
        nside = output_nside
        npix = hp.nside2npix(nside)

    skymodel = pyradiosky.SkyModel()
    skymodel.component_type = "healpix"
    skymodel.nside = nside
    skymodel.hpx_order = ordering
    skymodel.frame = frame_str
    skymodel.hpx_frame = frame_str
    skymodel.Nfreqs = 1
    skymodel.Ncomponents = npix
    skymodel.freq_array = Quantity(np.full(skymodel.Nfreqs, freq_mhz * 1e6), "hertz")
    skymodel.stokes = Quantity(
        np.zeros((4, skymodel.Nfreqs, skymodel.Ncomponents)),
        "Kelvin",  # Assume units of Kelvin
    )
    skymodel.stokes[0, 0, :] = map_data * units.Kelvin
    skymodel.hpx_inds = np.arange(skymodel.Ncomponents)
    skymodel.spectral_type = "subband"
    skymodel.history = history_str

    return skymodel


def downsample_healpix(input_map, output_nside, output_map_path=None, clobber=True):

    if isinstance(input_map, str):
        diffuse_map = pyradiosky.SkyModel()
        diffuse_map.read_skyh5(input_map)
    else:
        diffuse_map = input_map

    downsampled_map_data = hp.pixelfunc.ud_grade(
        diffuse_map.stokes[0, 0, :].value,
        output_nside,
        pess=True,
        order_in=diffuse_map.hpx_order,
    )
    diffuse_map.nside = output_nside
    diffuse_map.Ncomponents = hp.nside2npix(output_nside)
    diffuse_map.stokes = Quantity(
        np.zeros((4, diffuse_map.Nfreqs, diffuse_map.Ncomponents)), "Kelvin"
    )
    diffuse_map.stokes[0, 0, :] = downsampled_map_data * units.Kelvin
    diffuse_map.hpx_inds = np.arange(diffuse_map.Ncomponents)
    diffuse_map.check()

    if output_map_path is not None:
        diffuse_map.write_skyh5(
            output_map_path,
            run_check=True,
            clobber=clobber,
        )
    else:
        return diffuse_map


def concatenate_healpix_catalogs_in_frequency(catalog_list):

    if len(catalog_list) > 1:
        for cat_ind in range(len(catalog_list) - 1):
            if cat_ind == 0:
                cat1 = catalog_list[cat_ind]
            cat2 = catalog_list[cat_ind + 1]

            if cat1.nside != cat2.nside:
                if cat1.nside != np.min([cat1.nside, cat2.nside]):
                    cat1 = downsample_healpix(cat1, np.min([cat1.nside, cat2.nside]))
                if cat2.nside != np.min([cat1.nside, cat2.nside]):
                    cat2 = downsample_healpix(cat2, np.min([cat1.nside, cat2.nside]))
            if cat1.hpx_order != cat2.hpx_order:
                print("ERROR: Healpix ordering does not match.")
                sys.exit(1)
            if cat1.hpx_frame != cat2.hpx_frame:
                print("ERROR: Frame does not match.")
                sys.exit(1)

            skymodel = pyradiosky.SkyModel()
            skymodel.component_type = "healpix"
            skymodel.nside = cat1.nside
            skymodel.hpx_order = cat1.hpx_order
            skymodel.hpx_frame = cat1.hpx_frame
            skymodel.Nfreqs = cat1.Nfreqs + cat2.Nfreqs
            skymodel.Ncomponents = cat1.Ncomponents
            skymodel.freq_array = np.concatenate((cat1.freq_array, cat2.freq_array))
            skymodel.stokes = np.concatenate((cat1.stokes, cat2.stokes), axis=1)
            skymodel.hpx_inds = cat1.hpx_inds
            skymodel.spectral_type = "subband"
            skymodel.history = cat1.history

            skymodel.check()
            cat1 = skymodel

    return skymodel


if __name__ == "__main__":

    # Maps are downloaded from https://lambda.gsfc.nasa.gov/product/foreground/fg_ovrolwa_radio_maps_get.html

    filedir = "/lustre/rbyrne/skymodels"
    freqs_mhz = [
        "36.528",
        "41.760",
        "46.992",
        "52.224",
        "57.456",
        "62.688",
        "67.920",
        "73.152",
    ]
    for freq_ind, freq in enumerate(freqs_mhz):
        fits_filepath = f"{filedir}/ovro_lwa_sky_map_{freq}MHz.fits"
        skymodel_new = convert_fits_to_pyradiosky(
            fits_filepath,
            float(freq),
            output_frame="equatorial",
        )
        #skymodel_new = downsample_healpix(skymodel_new, 512)
        skymodel_new = downsample_healpix(skymodel_new, 32)
        if freq_ind == 0:
            skymodel = skymodel_new
        else:
            skymodel = concatenate_healpix_catalogs_in_frequency(
                [skymodel, skymodel_new]
            )
    skymodel.write_skyh5(
        f"{filedir}/ovro_lwa_sky_map_36-73MHz_nside32.skyh5",
        run_check=True,
        clobber=True,
    )
